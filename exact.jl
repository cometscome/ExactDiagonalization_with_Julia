#------------------------------------------------------
# Exact diagonalization code for the 2D Hubbard model
#                                  YN, Ph.D
#                                10/20/2017(mm/dd/yyyy)
#This might have bugs.
#This code is just for studying the ED method.
#
#
#------------------------------------------------------

module Exact
    import Combinatorics
    using SparseArrays
    using LinearAlgebra
    using IterativeSolvers

    export exact_main!



    function exact_main!(U,μ,nx,ny,β,tri_periodic_x,tri_periodic_y,fulldiag,nfix)

        nc = nx*ny
        nf = 4^(nc)
        nup = div(nc,2)
        ndown = div(nc,2)
        eps = 1e-6
        
        
    
        println("Exact diagonalizatoin code")
        println("Nx x Ny: ",nx," x ",ny)
    
        println( "U:",U)
        println( "μ:",μ)
        println( "Periodic boundary condision in x-direction:",tri_periodic_x)
        println( "Periodic boundary condision in y-direction:",tri_periodic_y)
        
    
    
        if fulldiag
            println( "----------------------------------------")
            println( "Dimension:",nf)
            println("Time for constructing operators:")
            @time (mat_cvec,mat_cdvec)=exact_init(nx,ny)   

            mat_h = const_h(nx,ny,nf,μ,U,mat_cvec,mat_cdvec,tri_periodic_x,tri_periodic_y)
            x = rand(nf)
            x = x/sqrt(x'*x)
            
            println("Time for calcualting eigenvalues:")
            @time r = lobpcg(mat_hf,false,1)
            λ = r.λ[1]
#            @time λ = lobpcg(mat_h,x,nf,eps)
            println("Minimum Eigenvalue: ",λ)

        end
    
    
        if nfix
            println( "----------------------------------------")
            println( "Numbers of each spin are fixed")
            println( "Num. of up spins: ",nup)
            println( "Num. of down spins: ",ndown)
        
            mup = binomial(nc, nup) 
            mdown = binomial(nc, ndown)
            mf = mup*mdown
            println(  "Dimension with fixed n:",mf)
            #println(μup)
            # println(μdown)
            println("Time for constructing operators:")
            @time mat_cvecf,mat_cdvecf=exact_init_fix2(nx,ny,nup,ndown,mup,mdown)
            mat_hf = const_h(nx,ny,mf,μ,U,mat_cvecf,mat_cdvecf,tri_periodic_x,tri_periodic_y)
            #println(mat_hf)
            x = rand(mf)
            x = x/sqrt(x'*x)
            
        
            println( "----------------------------------------")
            println( "Full dense matrix mode: ")
            println("Time for calcualting eigenvalues:")    
            @time λ= eigmin(Matrix(mat_hf))
            println("Minimum Eigenvalue: ",λ)
            println( "----------------------------------------")
            println( "Sparse matrix mode with the use of the LOBPCG: ")   
            println("Time for calcualting eigenvalues with:")       
            println("LOBPCG in IterativeSolvers")
            @time r = lobpcg(mat_hf,false,1)
            λ = r.λ[1]
            println("Minimum Eigenvalue: ",λ)  
            println("LOBPCG in this module")
            println("Time for calcualting eigenvalues with:")           
            @time λ = lobpcg_DIY(mat_hf,x,mf,eps)
            println("Minimum Eigenvalue: ",λ)
           

           

        
        
        end
    
    end

    function const_h(nx,ny,nn,μ,U,mat_cvec,mat_cdvec,tri_periodic_x,tri_periodic_y)
        mat_h = spzeros(nn,nn)
        mat_temp = spzeros(nn,nn)
        mat_temp2 = spzeros(nn,nn)
    
        for ix in 1:nx
            for iy in 1:ny
                for ispin in 1:2
                    isite = (ispin-1)*nx*ny+(iy-1)*nx + ix
                    jspin = ispin
                    jx = ix + 1
                    jy = iy
                    if tri_periodic_x
                        if jx > nx && nx != 1
                            jx += -nx
                        end
                    end
                    jsite = (jspin-1)*nx*ny+(jy-1)*nx + jx
                    if jx <= nx
                        v = -1.0
                        mat_c = mat_cvec[jsite]
                        mat_cdc = mat_cdvec[isite]
                        mat_h += v*mat_cdc*mat_c
                    end
                    
                    jx = ix - 1
                    jy = iy
                    if tri_periodic_x
                        if jx < 1 && nx != 1
                            jx += nx
                        end
                    end
                    jsite = (jspin-1)*nx*ny+(jy-1)*nx + jx
                    if jx > 0
                        v = -1.0
                        mat_c = mat_cvec[jsite]
                        mat_cdc = mat_cdvec[isite]
                        mat_h += v*mat_cdc*mat_c
                    end
                    
                    jx = ix
                    jy = iy + 1
                    if tri_periodic_y
                        if jy > ny && ny != 1
                            jy += -ny
                        end
                    end
                    jsite = (jspin-1)*nx*ny+(jy-1)*nx + jx
                    if jy <= ny
                        v = -1.0
                        mat_c = mat_cvec[jsite]
                        mat_cdc = mat_cdvec[isite]
                        mat_h += v*mat_cdc*mat_c
                    end   
                    
                    jx = ix 
                    jy = iy-1
                    if tri_periodic_y
                        if jy < 1 && ny != 1
                            jy += ny
                        end
                    end
                    jsite = (jspin-1)*nx*ny+(jy-1)*nx + jx
                    if jy > 0
                        v = -1.0
                        mat_c = mat_cvec[jsite]
                        mat_cdc = mat_cdvec[isite]
                        mat_h += v*mat_cdc*mat_c
                    end  
                        
                    jx = ix
                    jy = iy
                            
                    jsite = (jspin-1)*nx*ny+(jy-1)*nx + jx
                    v = -μ
                    mat_c = mat_cvec[jsite]
                    mat_cdc = mat_cdvec[isite]
                    mat_h += v*mat_cdc*mat_c
                      
                
                
                
                end
            end
        end
        
        for ix in 1:nx
            for iy in 1:ny
                ispin = 2
                isite = (ispin-1)*nx*ny+(iy-1)*nx + ix
                mat_c = mat_cvec[isite]
                mat_cdc = mat_cdvec[isite]
                mat_temp = mat_cdc*mat_c
                ispin = 1
                isite = (ispin-1)*nx*ny+(iy-1)*nx + ix
                mat_c = mat_cvec[isite]
                mat_cdc = mat_cdvec[isite]
                mat_temp2 = mat_cdc*mat_c
                mat_h += U*mat_temp2*mat_temp
            end
        end
            
    return mat_h
    end

    function exact_init(nx,ny)
        nc = nx*ny
        nf = 4^nc
        mat_cvec = []
        #println(mat_cvec)
        mat_cdvec = []
        mat_cdc = spzeros(nf,nf)
        for isite in 1:2nc
            mat_c = calc_matc(isite,nx,ny,0)
            #println(mat_c)
            push!(mat_cvec,mat_c)
            #println(mat_cvec)
            mat_cdc = mat_c'
            push!(mat_cdvec,mat_cdc)
        end
    
        return mat_cvec,mat_cdvec
        
    end

    function exact_init_fix2(nx,ny,nup,ndown,mup,mdown)
        nc = nx*ny
        mat_cvec = []
        #println(mat_cvec)
        mat_cdvec = []
        cn = [i for i in 1:nc]
        baseup = collect(Combinatorics.combinations(cn,nup)) #We generate combinations. nc_C_nup
        
        basedown = collect(Combinatorics.combinations(cn,ndown))
        basedown = map(x -> x .+nc,basedown)
        mf = mup*mdown
    
        p = 0
        for ispin in 1:2
            if ispin == 1
                targetbase = baseup
                otherbase = basedown
                nup2 = nup -1
                ndown2 = ndown
                annihilatebase = collect(Combinatorics.combinations(cn,nup-1))
            else
                targetbase = basedown
                otherbase = baseup
                nup2 = nup 
                ndown2 = ndown-1
                annihilatebase = collect(Combinatorics.combinations(cn,ndown-1))    
                annihilatebase=map(x -> x .+nc,annihilatebase)
            end
            mup2 = binomial(nc, nup2) 
            mdown2 = binomial(nc, ndown2) 
            mf2 = mup2*mdown2
        
            for isite in 1:nc
                ii = (ispin-1)*nx*ny+isite
                mat_c = calc_matc_fix(ii,ispin,targetbase,otherbase,annihilatebase,mf,mf2,p,mup,mup2,nc)
                push!(mat_cvec,mat_c)
                #println(mat_cvec)
                mat_cdc = mat_c'
                push!(mat_cdvec,mat_cdc)
            end
        end
    
        return mat_cvec,mat_cdvec
    
    end
    
    function calc_matc_fix(isite,ispin,targetbase,otherbase,annihilatebase,mf,mf2,p,mup,mup2,nc)
        mat_c = spzeros(mf2,mf)
        i = 0
        for basis in targetbase
            if isite in basis
                j = 0
                for basis2 in otherbase
                    #println("basis:",basis)
                    #println("basis2:",basis2)
                    vec_i = calc_veci(basis,basis2,nc)
                    #println("vec_i:",vec_i)
                    vec_iout,sig = calc_c_cd(isite,vec_i,p,nc)                    
                    if sig != 0
                        #println("vec_iout:",vec_iout)
                        b3 = calc_basis(ispin,vec_iout,nc)
                        #println(annihilatebase)
#                        println("b3:",b3)
                        inann = findnext(x->x==b3,annihilatebase,1)
#                        inann = findfirst(annihilatebase,b3)
                        inann += -1
                        #println(inann)
                        if ispin == 1
                            ih = mup2*j + inann
                            jh = mup*j + i
                        else
                            ih = mup2*inann + j
                            jh = mup*i + j
                        end
                        #println(ih,"\t",jh)
                        mat_c[ih+1,jh+1] = sig
                    end
                    j += 1
            
                end
            
            end
            i += 1
        end
        return mat_c
    end

    function calc_basis(ispin,vec_i,nc)
        basis = Int64[]
        for i in 1:nc
            j = vec_i[i+(ispin-1)*nc]
            if j != 0
                push!(basis,i*j+(ispin-1)*nc)
            end
        end
        return basis
    end
    
    function calc_veci(basis,basis2,nc)
        vec_i = zeros(Int64,2nc)
        for i in basis
            vec_i[i] = 1
        end
        for i in basis2
            vec_i[i] = 1
        end
    
        return vec_i
    end

    function calc_matc(isite,nx,ny,p)
        nc = nx*ny
        nf = 4^nc
        mat_c = spzeros(nf,nf)
        for jj in 1:nf
            vec_i = calc_ii2vec(jj,nc)
            vec_iout,sig = calc_c_cd(isite,vec_i,p,nc)
            if sig != 0
                ii = calc_vec2ii(vec_iout,nc)
                mat_c[ii,jj] = sig
            end
        end
    
        return mat_c
        
    end

    function calc_c_cd(isite,vec_i,p,nc)
        vec_iout = vec_i
        sig = calc_sign(isite,vec_i,p,nc)
        if sig == 0
            vec_iout[:] = -1
        else
            vec_iout[isite] = p
        end
    
        return vec_iout,sig
    end

    function calc_sign(isite,vec_i,p,nc)
        if vec_i[isite] == p
            sig = 0
        else
            sig = 1
            isum = sum(vec_i[isite+1:2nc])
            sig = (-1)^(isum)
        end
        return sig
    end

    function calc_vec2ii(vec_iout,nc)
        ii = 1
        for isite in 1:2nc
            ii += vec_iout[isite]*(2^(isite-1))
        end
        return ii
    end

    function calc_ii2vec(ii,nc)
        vec_i = zeros(Int64,2nc)
        iii = ii-1
        vec_i[1]=(iii)%2
        #println("iii\t",vec_i[1])
        iii = div(ii-vec_i[1],2)
        #println(iii)
        for i in 2:2nc
            #println(iii%2)
            vec_i[i] = iii%2
            iii = div(iii-vec_i[i],2)
        end
    
        return vec_i
    end


    function lobpcg_DIY(A,x0,n,eps)
        #println("n",n)
        x = ones(Float64,n)
        x = x/sqrt(dot(x,x))
        Ax = A*x0
        λ = x0'*Ax
        r = Ax - x0*λ
        p = zeros(Float64,n)
        z = zeros(Float64,n,3)
        #println(z[n,3])
        #println(typeof(z))
        ztemp = zeros(Float64,n,3)
        zhz = zeros(Float64,3,3)
        zv = zeros(Float64,3,3)
        v = zeros(Float64,3,3)
        zλ = zeros(Float64,3)
    
        itemax = 100000
        for ite in 1:itemax
            z[:,1] = x[:]
            z[:,2] = r[:]
            z[1:n,3] = p[1:n]
            if ite == 1
                nz = 2
            else
                nz = 3
            end
            ztemp = gram(n,nz,z[:,1:nz])
            z[1:n,1:nz] = A[1:n,1:n]*ztemp[1:n,1:nz]

            zhz[1:nz,1:nz] = ztemp[:,1:nz]'*A[:,:]*ztemp[:,1:nz]#ztemp[:,1:nz]'*z[:,1:nz]
            zhz[1:nz,1:nz] = (zhz[1:nz,1:nz]+zhz[1:nz,1:nz]')/2
            (zλ[1:nz],zv[1:nz,1:nz]) = eigen(zhz[1:nz,1:nz])
            v[1:nz,1] = zv[1:nz,1]

            λ = zλ[1]
            x[:] = ztemp[:,1:nz]*zv[1:nz,1]
            r = x*λ        
            Ax = A*x
            r = Ax -r
            reps = sqrt(dot(r,r))
            if eps > reps 
                #println(ite,"\t",reps)
                norm = sqrt(dot(x,x))
                x = x/norm
                #println(λ)
                #println(sum(λ*x-A*x))
                break
            end
        
            if ite % 1000 == 0
                            #println(zλ[1:nz])
                #println(ite,"\t",reps,"\t",λ)
            
            end
            if ite == 1
                p = r[:]*v[2,1]
            else
                p = r*v[2,1] + p*v[3,1]
            end
        
        end
        
    
        
    
        return λ
    end

    function gram(m,n,mat_v)
        mat_v_out = mat_v
        viold = zeros(Float64,m)
        i = 1
        j = 1
        for i in 1:n
            viold = mat_v_out[:,i]
            if i > 1
                for j in 1:i-1
                    nai = dot(mat_v_out[1:m,j],viold[1:m])
                    vi = viold - nai*mat_v_out[:,j]
                    viold = vi
                end
            end
            norm = sqrt(dot(viold,viold))
            mat_v_out[:,i] = viold/norm
        end
        return mat_v_out
    end




end

