abstract type AbstractGame end 
abstract type AbstractPlayer end
abstract type AbstractMove end

const PLAYER1 = 1
const PLAYER2 = 2

include("montecarlo.jl")
include("TicTacToe.jl")
include("Connect4.jl")
include("pit.jl")
include("EGo.jl")
