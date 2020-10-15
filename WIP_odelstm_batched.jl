using Flux
using LinearAlgebra
using OrdinaryDiffEq
using DiffEqBase
using Zygote
using DiffEqSensitivity
using CSV
using Dates
using Distributed
using LoopVectorization

import Flux: Recur, LSTM, RNN, LSTMCell, RNNCell, glorot_uniform, @functor, hidden, reset!, @epochs


function load_person_data()
    csv = CSV.File("ConfLongDemo_JSI.txt"; delim=',', header=false)

    s_ids = Dict("010-000-024-033"=>1,
                 "010-000-030-096"=>2,
                 "020-000-032-221"=>3,
                 "020-000-033-111"=>4)

    cls_ids = Dict("lying down"=>1,
                   "lying"=>1,
                   "sitting down"=>2,
                   "sitting"=>2,
                   "standing up from lying"=>3,
                   "standing up from sitting"=>3,
                   "standing up from sitting on the ground"=>3,
                   "walking"=>4,
                   "falling"=>5,
                   "on all fours"=>6,
                   "sitting on the ground"=>7,)

    all_seqs = []
    all_cls_seqs = []
    seq = []
    cls_seq = []
    prev_l = nothing

    for (i, l) in enumerate(csv)
        p = l[1]
        if i == 1602 || i == 1603 || i == 1604 || i == 161790 || i == 161791
            #@show l
        end
        sensor = s_ids[l[2]]
        s_onehot = Flux.onehot(sensor, collect(1:maximum(values(s_ids))))
        x,y,z = l[5],l[6],l[7]
        cls = cls_ids[l[8]]

        if prev_l != nothing && (p != prev_l[1])
            # new sequence
            push!(all_seqs, seq)
            push!(all_cls_seqs, cls_seq)

            prev_l = l
            seq = []
            cls_seq = []
        end

        if prev_l == nothing
            lag = 2 #/ 1e3
        else
            lag = (l[3] - prev_l[3]) / 1e5
            #@show lag
        end

        if lag == 0
            #println("HMMM")
            #@show lag
            #@show i
            lag = 1
        elseif lag < 0
            #println("OH NO")
            #@show lag
            #@show i
            lag = 1
        end

        if lag > 100
            @show i
        end


        prev_l = l
        push!(seq, (Float32.([s_onehot; [x,y,z]]), Float32(lag)))
        push!(cls_seq, cls)
    end

    all_seqs = Vector{Vector{Tuple{Vector{Float32},Float32}}}(all_seqs)
    all_cls_seqs = Vector{Vector{Float32}}(all_cls_seqs)

    train_xs = all_seqs[1:5]
    train_ys = all_cls_seqs[1:5]
    test_xs = all_seqs#[201:300]
    test_ys = all_cls_seqs#[201:300]



    train_loader = Flux.Data.DataLoader((train_xs,train_ys), batchsize=32, shuffle=true, partial=false)
    train_loader_f = Flux.Data.DataLoader((train_xs,train_ys), batchsize=length(train_xs), shuffle=false, partial=false)


    x = []
    xs_seq = []
    xs = zeros(7,32)
    ts = zeros(32)
    for (i,seq) in enumerate(train_xs)
        bi = 1
        for (j,inst) in enumerate(seq)
            xs[:,bi] = inst[1]
            ts[bi] = inst[2]
            bi += 1
            if bi == 33
                #@show size(xs)
                #@show size(ts)
                push!(xs_seq, (xs,ts))
                bi = 1
            end
        end
        push!(x, xs_seq)
        xs_seq = []
    end

    y = []
    ys_seq = []
    ys = zeros(32)
    for (i,seq) in enumerate(first(train_loader_f)[2])
        bi = 1
        for (j,inst) in enumerate(seq)
            ys[bi] = inst
            if bi == 33
                push!(ys_seq, ys)
                bi = 1
            end
        end
        push!(y, ys_seq)
        ys_seq = []
    end

    # [rand(8,32) for j in 1:100]
    # [(rand(7,32),rand(32)) for j in 1:100]

    batchsize = 32
    seq_len = 64
    num_batches = 100

    x = [[(rand(Float32,7,batchsize), 1 .+ 100 .* rand(Float32,batchsize)) for j in 1:seq_len] for i in 1:num_batches]
    y = [[rand(collect(1:7),batchsize) for j in 1:seq_len] for i in 1:num_batches]

    # @show size(x[1][1][1])
    # @show size(x[1][1][2])
    # @show size(y[1][1])

    test_loader_f = Flux.Data.DataLoader((test_xs,test_ys), batchsize=length(test_xs), shuffle=false, partial=false)
    return train_loader, train_loader_f, test_loader_f, x,y
end


mutable struct CTRNNCellB{V}
    rnn::RNNCell
    τ::V
    #t::Float32
end

# ####
# TODO: vector of τ or scalar?
# ####
function CTRNNCellB(in::Integer, out::Integer, τ=Float32[1 for i in 1:out];
                    init = glorot_uniform)
    rnn = RNNCell(in, out, init=init)
    #t = Float32(0)
    CTRNNCellB(rnn, τ)
end

function (m::CTRNNCellB)(h, x)
    xib, tib = x
    batchsize = length(tib)
    #dudt(u,p,t) = -p .* u
    function dudt!(du,u,p,t)
        du .= -p .* u
        nothing
    end


    # if h isa Vector
    #     hnew = Zygote.Buffer(h, length(h), batchsize)
    #     # first call
    #     for i in 1:batchsize
    #         for j in 1:length(h)
    #             hnew[j,i] = h[j]
    #         end
    #     end
    #     h = copy(hnew)
    # end
    buf = Zygote.Buffer(h, size(h,1), batchsize)
    #@show size(buf)


    odef = ODEFunction{true}(dudt!)
    #@show size(h)
    prob = ODEProblem{true}(odef, h[:,1], Float32.((0, 100)), m.τ)


    alg = VCABM()
    #alg = Tsit5()
    #alg = TRBDF2()
    sense = QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))
    #sense = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))
    #sense = InterpolatingAdjoint(autojacvec=false)
    sense = ForwardDiffSensitivity()

    function prob_func(prob,i,repeat)
        tspan = Float32.((0, tib[i]))
        #@show tspan
        #remake(prob, u0=h[:,i], tspan=tspan)
        ODEProblem{true}(odef, h[:,i], tspan, m.τ)
    end
    ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)

    ensemble_sol = solve(ensemble_prob, alg, EnsembleSerial(),
                         trajectories = batchsize,
                         save_start = false, save_everystep = false,
                         sensealg = sense)

    #ensemble_sol = []
    p_sol = Zygote.Buffer(h, size(h,1), batchsize)

    function solve_instance(i)
        prob = ODEProblem{true}(odef, h[:,i], Float32.((0, tib[i])), m.τ)
        sol = Array(solve(prob,alg,save_start=false,save_everystep=false,sensealg=sense))[:,1]

        for j in 1:length(sol)
            p_sol[j,i] = sol[j]
        end
    end
    #vsol = []
    #vmapt(solve_instance, collect(1:batchsize))
    #pmap((i)->solve_instance(i), 1:batchsize)
    #vmap(solve_instance,collect(1:batchsize))

    #Threads.@threads for i in 1:batchsize
    # for i in 1:batchsize
    #     prob = ODEProblem{true}(odef, h[:,i], Float32.((0, tib[i])), m.τ)
    #     sol = Array(solve(prob,alg,save_start=false,save_everystep=false,sensealg=sense))[:,1]
    #
    #     for j in 1:length(sol)
    #         p_sol[j,i] = sol[j]
    #     end
    #     # #push!(ensemble_sol, sol)
    #     # if length(ensemble_sol) == 0
    #     #     ensemble_sol = [sol]
    #     # else
    #     #     ensemble_sol = [ensemble_sol, sol]
    #     # end
    #
    # end

    #
    #hi′ = [Array(ensemble_sol[i])[:,1] for i in 1:batchsize]
    for i in 1:batchsize
        sol = Array(ensemble_sol[i])[:,1]
        for j in 1:length(sol)
            buf[j, i] = sol[j]
        end
    end
    hi′ = copy(buf)
    #hi′ = copy(p_sol)
    #hi′ = Array(solve(prob,alg,save_start=false,save_everystep=false,sensealg=sense))[:,1]
    hi = m.rnn(hi′, xib)[1] # ret of RNNCell: (h, h)
    #@show hi
    hi, hi
end

# function (m::CTRNNCellB)(h, x)
#     xi, ti = x
#     #dudt(u,p,t) = -p .* u
#     function dudt!(du,u,p,t)
#         du .= -p .* u
#         nothing
#     end
#
#     odef = ODEFunction{true}(dudt!)
#     prob = ODEProblem{true}(odef, h, (Float32(0), ti), m.τ)
#
#     alg = VCABM()
#     #alg = Tsit5()
#     #alg = TRBDF2()
#     sense = QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))
#     #sense = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))
#     #sense = InterpolatingAdjoint(autojacvec=false)
#
#     hi′ = Array(solve(prob,alg,save_start=false,save_everystep=false,sensealg=sense))[:,1]
#     hi = m.rnn(hi′, xi)[1] # ret of RNNCell: (h, h)
#     hi, hi
# end

Flux.@functor CTRNNCellB
hidden(m::CTRNNCellB) = (m.rnn.h)
Flux.trainable(m::CTRNNCellB) = (Flux.trainable(m.rnn), m.τ)




mutable struct ODELSTMCellB
    lstm::LSTMCell
    odernn::CTRNNCellB
end

function ODELSTMCellB(in::Integer, ldim::Integer, out::Integer; init = glorot_uniform)
    lstm = LSTMCell(in, ldim, init=init)
    odernn = CTRNNCellB(ldim, out, init=init)
    ODELSTMCellB(lstm, odernn)
end


@functor ODELSTMCellB
Flux.hidden(m::ODELSTMCellB) = (m.lstm.h, m.lstm.c)
#Flux.trainable(m::ODELSTMCellB) = (Flux.trainable(m.lstm), Flux.trainable(m.odernn), m.Wo, m.bo)
Flux.trainable(m::ODELSTMCellB) = (Flux.trainable(m.lstm), Flux.trainable(m.odernn))


function (m::ODELSTMCellB)((h, c), x)
    xi, ti = x
    batchsize = length(ti)

    # if h isa Vector
    #     hnew = Zygote.Buffer(h, length(h), batchsize)
    #     cnew = Zygote.Buffer(c, length(c), batchsize)
    #     # first call
    #     for i in 1:batchsize
    #         for j in 1:length(h)
    #             hnew[j,i] = h[j]
    #             cnew[j,i] = c[j]
    #         end
    #     end
    #     h = copy(hnew)
    #     c = copy(cnew)
    # end

    (hi′, ci) = m.lstm((h, c), xi)[1] # ret of LSTMCell: (h, c), h
    hi = m.odernn(hi′, (h,ti))[2] # ret of CTRNNCell: (h, t), h
    return (hi, ci), hi
end



#rand_x = [(rand(Float32,7,32), abs.(10 .+ 100 .* rand(Float32,32))) for j in 1:100]
rand_x = load_person_data()[4]
#push!(rand_x, (rand(Float32,7,30), abs.(100 .* rand(Float32,30))))
rand_m = Chain(Recur(ODELSTMCellB(7,20,20)),Dense(20,7),softmax)
rand_m(rand_x[1][1])
rand_m(rand_x[1][2])
rand_m(rand_x[1][end])
Flux.reset!(rand_m)
@time rand_o = rand_m.(rand_x[1])


function lossf(x, y, model)
    #@show size(x)
    seq_len = length(x)
    classes = collect(1:7)
    #@show seq_len
    #@show size(x[1][1])

    ŷ = model.(x)
    #@show length(ŷ)
    ŷ = ŷ[seq_len]
    y_onehot = Flux.onehotbatch(y[end], classes)
    l::Float32 = Flux.Losses.crossentropy(ŷ, y_onehot)
    Flux.reset!(model)
    l
end

function start_opt(epochs)

    train_loader, train_loader_f, test_loader_f, xs, ys = load_person_data()

    model = Chain(Recur(ODELSTMCellB(7,10,10)), Dense(10,7), softmax)
    #model = Chain(Recur(CTRNNCellB(7,10)), Dense(10,7), softmax)

    # @time lossf(tx1,ty1)
    # @time Zygote.gradient(x->lossf(x..., model), (tx1,ty1))
    # @time Zygote.gradient(x->lossf(x..., model), (tx1,ty1))

    zdata = zip(xs,ys)
    function evalcb(data)
        sum([lossf(x,y,model) for (x,y) in data])
        # l = 0.0
        # for (x,y) in data
        #     l += lossf(x,y,model)
        # end
        # l
    end
    #evalcb() = @show sum(lossf(first(zdata)..., mode]))
    @show evalcb([(xs[i],ys[i]) for i in 1:length(xs)])

    ps = Flux.params(model)

    for i in 1:epochs
        @info "Epoch $i"
        #Flux.train!((x,y)->lossf(x,y,model), ps, train_loader, ADAM(0.002))
        #for j in 1:length(xs)
        Flux.train!((x,y)->lossf(x,y,model), ps, [(xs[1],ys[1])], ADAM(0.002))
        #end
        train_loss = evalcb([(xs[i],ys[i]) for i in 1:length(xs)])
        test_loss = evalcb([(xs[i],ys[i]) for i in 1:length(xs)])
        @show train_loss, test_loss
    end
    ps
end

ps = start_opt(1)
#@time ps = start_opt(50)

