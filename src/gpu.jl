
abstract type AbstractDevice end 
abstract type AbstractGPUDevice <: AbstractDevice end 
struct DeviceCPU <: AbstractDevice end 
struct DeviceCUDA <: AbstractGPUDevice end 

"""
    DetermineDevice(; gpu::Union{Nothing, Bool}=nothing)   

Initializes the device that is used. Returns either `DeviceCPU` or `DeviceCUDA`. If no `gpu` keyword argument is given, it determines automatically if a GPU is available.
"""
function DetermineDevice(; gpu::Union{Nothing, Bool}=nothing)   
    if isnothing(gpu)
        dev = CUDA.functional() ? DeviceCUDA() : DeviceCPU()
    else 
        dev = gpu ? DeviceCUDA() : DeviceCPU()
    end 
    return dev 
end 

isgpu(::DeviceCUDA) = true 
isgpu(::DeviceCPU) = false

DeviceArray(dev::DeviceCUDA, x) = adapt(CuArray, x)
DeviceArray(dev::DeviceCPU, x) = adapt(Array, x)

FullDeviceArray(dev::DeviceCUDA, x) = CuArray(x)
FullDeviceArray(dev::DeviceCPU, x) = Array(x)

gpu(dev::DeviceCUDA, x) = cu(x)
gpu(dev::DeviceCPU, x) = x
