# Load & initialize CUDA driver

const CUDA_LIB = @windows? "nvcuda.dll" : "libcuda"
dlopen(CUDA_LIB)	# loads library, throws an error if not found

macro cucall(fv, argtypes, args...)
    f = eval(fv)
    quote
        _curet = ccall( ($(Meta.quot(f)), "libcuda"), Cint, $argtypes, $(args...) )
        if _curet != 0
            throw(CuDriverError(int(_curet)))
        end
    end
end

function initialize()
    @cucall(cuInit, (Cint,), 0)
    println("CUDA Driver Initialized")
end

initialize()


# Get driver version

function driver_version()
    a = Cint[0]
    @cucall(cuDriverGetVersion, (Ptr{Cint},), a)
    return int(a[1])
end

const DriverVersion = driver_version()

if DriverVersion < 4000
    error("CUDA of version 4.0 or above is required.")
end


# box a variable into array

cubox{T}(x::T) = T[x]

