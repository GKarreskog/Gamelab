using BenchmarkTools
import Printf: @printf, @sprintf
using NLsolve


struct Game
    row::Matrix{Real}
    col::Matrix{Real}
end


Base.transpose(g::Game) = Game(transpose(g.col), transpose(g.row))
Base.size(g::Game) = size(g.row)[1]

Base.show(io::IO, g::Game) = begin
    # pritnln(g.name)
    for i in 1:size(g)
        for j in 1:size(transpose(g))
            # print(Int(g.row[i,j]), ",", Int(g.col[i,j]), "  ")
            @printf("%3.0f , %3.0f | ", g.row[i,j], g.col[i,j])
        end
        println()
    end
end

function softmax(x)
    ex = exp.(x)
    ex ./= sum(ex)
    ex
end

function logit_best_reply(p_opp, λ, g::Game)
    softmax(λ .* (g.row * p_opp))[:]
end


function mean_square(x, y)
    sum((x - y).^2)
end

# function find_qre(λ, g::Game, tol=0.01)
#     opp_g = transpose(g)
#     p1 = ones(size(g))/size(g)
#     p2 = ones(size(opp_g))/size(opp_g)
#     logit_p1 = logit_best_reply(p2, λ, g)
#     logit_p2 = logit_best_reply(p1, λ, opp_g)
#     while (mean_square(p1, logit_p1) > tol) || (mean_square(p2, logit_p2) > tol)
#         p1 = logit_p1
#         p2 = logit_p2
#         logit_p1 = logit_best_reply(p2, λ, g)
#         logit_p2 = logit_best_reply(p1, λ, opp_g)
#     end
#     return (logit_p1, logit_p2)
# end

function find_qre(λ, g::Game; init_x = nothing)
    opp_g = transpose(g)
    function f(x)
        p1 = x[1:size(g)]
        p2 = x[(size(g)+1):end]
        log_p1 = logit_best_reply(p2, λ, g)
        log_p2 = logit_best_reply(p1, λ, opp_g)
        opp_g
        return vcat(log_p1, log_p2)
    end
    if init_x == nothing
        init_p1 = ones(size(g))/size(g)
        init_p2 = ones(size(opp_g))/size(opp_g)
        init_x = vcat(init_p1, init_p2)
    end
    res = fixedpoint(f, init_x)
    (res.zero[1:size(g)], res.zero[(size(g)+1):end])
end

g = game3
λs = 0.1:0.1:5.
p1s = []
p2s = []
for λ in λs
    p1, p2 = find_qre(λ, g; init_x=[0.1, 0.4, 0.4, 0.1, 0.1, 0.8, 0.1])
    push!(p1s, p1)
    push!(p2s, p2)
end

p1s
p2s

rps = Game([[10 0 11]; [12 10 5]; [0 12 10]], [[10 12 0]; [0 10 12]; [11 5 10]])
cent = Game([[2 2 2 2 2]; [1 4 4 4 4]; [1 3 10 10 10]; [1 3 5 18 18];[1 3 5 7 30]], [[0 0 0 0 0]; [3 1 1 1 1]; [3 7 4 4 4]; [3 7 13 6 6]; [3 7 13 23 8]])
game3 = Game([[12 4 0]; [4 12 0]; [0 14 2]; [6 6 6]], [[12 4 14]; [4 12 0]; [0 0 2]; [0 0 0]])

g = rps
find_qre(1., rps)

logit_best_reply(p_opp, 0.5, rps)

p = [0.1, 0.5, 0.4]
p_opp = [0.3, 0.5, 0.2]
softmax(p_opp' * rps.row)[:]
