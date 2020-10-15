using Flux
using LinearAlgebra
using OrdinaryDiffEq
using Zygote
using DiffEqSensitivity
using CSV
using Dates

import Flux: Recur, LSTM, RNN, LSTMCell, RNNCell, glorot_uniform, @functor, hidden, reset!, @epochs


function load_person_data()
    csv = CSV.File("ConfLongDemo_JSI.txt"; delim=',', header=false)

    s_ids = Dict("010-000-024-033"=>1,
                 "010-000-030-096"=>2,
                 "020-000-032-221"=>3,
                 "020-000-033-111"=>4)

    cls_ids = Dict(
        "lying down"=>0,
        "lying"=>0,
        "sitting down"=>1,
        "sitting"=>1,
        "standing up from lying"=>2,
        "standing up from sitting"=>2,
        "standing up from sitting on the ground"=>2,
        "walking"=>3,
        "falling"=>4,
        "on all fours"=>5,
        "sitting on the ground"=>6,
    )

    prev_l = nothing
    seq = []
    all_seqs = []
    cls_seq = []
    all_cls_seqs = []
    df = DateFormat("dd.mm.yyyy H:M:S:s")
    for (i, l) in enumerate(csv)
        p = l[1]
        sensor = s_ids[l[2]]
        s_onehot = Flux.onehot(sensor, collect(1:maximum(values(s_ids))))
        millis = l[3]
        dt = DateTime(l[4], df)
        x,y,z = l[5],l[6],l[7]
        cls = cls_ids[l[8]]

        if prev_l != nothing && (p != prev_l[1] || cls != cls_ids[prev_l[8]])
            # new sequence
            if prev_l != nothing
                push!(all_seqs, seq)
                push!(all_cls_seqs, cls_seq)
            end

            prev_l = l
            seq = []
            cls_seq = []
        end

        if prev_l == nothing
            lag = 20 #/ 1e3
        else
            lag = dt.instant.periods.value - DateTime(prev_l[4], df).instant.periods.value # millis
            #lag = lag / 1e3
        end
        if lag == 0
            lag = l[3] - prev_l[3]
        end
        prev_l = l
        if lag <= 0
            continue
            @show i
            println("OH NO")
        end

        if lag > 1000
            @show i
        end


        push!(seq, (Float32.([s_onehot; [x,y,z]]), Float32(lag)))
        push!(cls_seq, cls)
    end

    all_seqs = Vector{Vector{Tuple{Vector{Float32},Float32}}}(all_seqs)
    all_cls_seqs = Vector{Vector{Float32}}(all_cls_seqs)

    train_xs = all_seqs[1:200]
    train_ys = all_cls_seqs[1:200]
    test_xs = all_seqs[201:300]
    test_ys = all_cls_seqs[201:300]

    train_loader = Flux.Data.DataLoader((train_xs,train_ys), batchsize=32, shuffle=true, partial=false)
    train_loader_f = Flux.Data.DataLoader((train_xs,train_ys), batchsize=length(train_xs), shuffle=false, partial=false)
    test_loader_f = Flux.Data.DataLoader((test_xs,test_ys), batchsize=length(test_xs), shuffle=false, partial=false)




    ### RANDOM DATA

    seq_len = 100
    num_seqs = 1000
    x = [[(rand(Float32,7), 1 .+ 100 .* rand(Float32)) for j in 1:seq_len] for i in 1:num_seqs]
    y = [[rand([0,1,2,3,4,5,6]) for j in 1:seq_len] for i in 1:num_seqs]

    return train_loader, train_loader_f, test_loader_f, x, y
end


mutable struct CTRNNCell1{V}
    rnn::RNNCell
    τ::V
    #t::Float32
end

# ####
# TODO: vector of τ or scalar?
# ####
function CTRNNCell1(in::Integer, out::Integer, τ=Float32[1 for i in 1:out];
                    init = glorot_uniform)
    rnn = RNNCell(in, out, init=init)
    #t = Float32(0)
    CTRNNCell1(rnn, τ)
end

function (m::CTRNNCell1)(h, x)
    xi, ti = x
    #dudt(u,p,t) = -p .* u
    function dudt!(du,u,p,t)
        du .= -p .* u
        nothing
    end
    odef = ODEFunction{true}(dudt!)
    prob = ODEProblem{true}(odef, h, Float32.((0, ti)), m.τ)

    alg = VCABM()
    #alg = Tsit5()
    #alg = TRBDF2()
    sense = QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))
    #sense = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))
    #sense = InterpolatingAdjoint(autojacvec=false)

    hi′ = Array(solve(prob,alg,save_start=false,save_everystep=false,sensealg=sense))[:,1]
    hi = m.rnn(hi′, xi)[1] # ret of RNNCell: (h, h)
    hi, hi
end

Flux.@functor CTRNNCell1
hidden(m::CTRNNCell1) = (m.rnn.h)
Flux.trainable(m::CTRNNCell1) = (Flux.trainable(m.rnn), m.τ)



mutable struct ODELSTMCell1
    lstm::LSTMCell
    odernn::CTRNNCell1
end

function ODELSTMCell1(in::Integer, ldim::Integer, out::Integer; init = glorot_uniform)
    lstm = LSTMCell(in, ldim, init=init)
    odernn = CTRNNCell1(ldim, out, init=init)
    ODELSTMCell1(lstm, odernn)
end


@functor ODELSTMCell1
Flux.hidden(m::ODELSTMCell1) = (m.lstm.h, m.lstm.c)
#Flux.trainable(m::ODELSTMCell1) = (Flux.trainable(m.lstm), Flux.trainable(m.odernn), m.Wo, m.bo)
Flux.trainable(m::ODELSTMCell1) = (Flux.trainable(m.lstm), Flux.trainable(m.odernn))

function (m::ODELSTMCell1)((h, c), x)
    xi, ti = x
    (hi′, ci) = m.lstm((h, c), xi)[1] # ret of LSTMCell: (h, c), h
    #hi = m.odernn(hi′, (h,ti))[2] # ret of CTRNNCell: (h, t), h
    hi = m.odernn(h, (hi′,ti))[2] # ret of CTRNNCell: (h, t), h
    #oi = m.Wo*hi .+ m.bo # TODO: return odernn out? https://github.com/mlech26l/learning-long-term-irregular-ts/blob/aec908c186a64b2f3af17bbb1b73e71d4640bd4a/node_cell.py#L265
    #return (hi, ci, ti), oi
    return (hi, ci), hi
end

function lossf(x, y, model)
    classes = collect(0:6)
    ŷ = model.(x)[end]
    y_onehot = Flux.onehot(y[end], classes)
    l = Flux.Losses.crossentropy(ŷ, y_onehot)
    Flux.reset!(model)
    l
end

function start_opt(epochs)

    train_loader, train_loader_f, test_loader_f, x, y = load_person_data()

    model = Chain(Recur(ODELSTMCell1(7,10,10)), Dense(10,7), softmax)

    # @time lossf(tx1,ty1)
    # @time Zygote.gradient(x->lossf(x..., model), (tx1,ty1))
    # @time Zygote.gradient(x->lossf(x..., model), (tx1,ty1))

    evalcb() = @show lossf(x[1],y[1], model)
    evalcb()

    ps = Flux.params(model)

    for i in 1:epochs
        @info "Epoch $i"
        Flux.train!((x,y)->lossf(x,y,model), ps, [(x[j],y[j]) for j in 1:length(x)], ADAM(0.002))
        train_loss = lossf(x[1],y[1], model)
        #test_loss = lossf(x[1],y[1], model)
        @show train_loss#, test_loss
    end
    ps
end

@time ps = start_opt(1)
@time ps = start_opt(50)

