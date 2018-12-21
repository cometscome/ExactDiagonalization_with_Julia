include("exact.jl")
using .Exact
const U=-2
const μ=U/2
const nx = 4
const ny = 2
const β=100.0
const tri_periodic_x = false
const tri_periodic_y = false
const fulldiag = false
const nfix = true

exact_main!(U,μ,nx,ny,β,tri_periodic_x,tri_periodic_y,fulldiag,nfix)