
using Lux, NNlib

"""
    ConvBlock(;kernel=(3,3), N_channels::Pair=12=>12, activation=relu, conv_type=LongLatConv) 

Default Convulution block used in the [`UNet`](@ref), currently three convolutational layers. `conv_type` has to be either `Lux.Conv` or a function that follow the same syntax.  
"""
ConvBlock(;kernel=(3,3), N_channels::Pair=12=>12, activation=relu) = FlatChain(Lux.Conv(kernel, N_channels[1]=>N_channels[2], activation; pad=Lux.SamePad()), Lux.Conv(kernel, N_channels[2]=>N_channels[2], activation; pad=Lux.SamePad()), Lux.Conv(kernel, N_channels[2]=>N_channels[2], activation; pad=Lux.SamePad()))

"""
    LongLatConv(k::NTuple, ch::Pair{<:Integer,<:Integer},
activation=identity; dilation=1, stride=1, kwargs...) 

Wraps around `Lux.Conv` but with periodic padding along the longitude and same padding along the latitude dimension. 
"""
struct LongLatConv{T} <: Lux.AbstractLuxWrapperLayer{:conv}
    conv::T 
end

(m::LongLatConv)(x,ps,st) = m.conv(x,ps,st)

function LongLatConv(k::NTuple, ch::Pair{<:Integer,<:Integer},
activation=identity; dilation=1, stride=1, kwargs...) 
    
    N_pads = Lux.calc_padding(Lux.SamePad(), k, dilation, stride)

    LongLatConv(FlatChain(WrappedFunction(x -> NNlib.pad_circular(x, (0, 0, N_pads[3], N_pads[4]))), Conv(k, ch, activation; pad=(N_pads[1],N_pads[2],0,0), stride, dilation, kwargs...)))
end 

LongLatConvBlock(;kernel=(3,3), N_channels::Pair=12=>12, activation=relu, conv_type=LongLatConv) = FlatChain(conv_type(kernel, N_channels[1]=>N_channels[2], activation), conv_type(kernel, N_channels[2]=>N_channels[2], activation), conv_type(kernel, N_channels[2]=>N_channels[2], activation))

"""
    LongLatConvTranspose(k::NTuple, ch::Pair{<:Integer,<:Integer},
activation=identity; dilation=1, stride=1, kwargs...) 

Wraps around `Lux.ConvTranspose` but with periodic padding along the longitude and same padding along the latitude dimension. 
"""
function LongLatConvTranspose(k::NTuple, ch::Pair{<:Integer,<:Integer},
activation=identity; dilation=1, stride=1, kwargs...) 
    
    N_pads = Lux.calc_padding(Lux.SamePad(), k, dilation, stride)

    FlatChain(WrappedFunction(x -> NNlib.pad_circular(x, (0, 0, N_pads[3], N_pads[4]))), ConvTranspose(k, ch, activation; pad=(N_pads[1],N_pads[2],0,0), stride, dilation, kwargs...))
end 

"""
    UNet(; kernel=(3,3), N_channels=3, activation=NNlib.relu, downconv=DownConvBlock, upconv=UpConvBlock, upsample= () -> Lux.Upsample(2), pool = () -> Lux.MaxPool((2,2))) <: Lux.AbstractLuxWrapperLayer{:model}

Initializes a 4-level UNet ANN with the specified hyperparameters. `downconv` and `upconv` are receiving the kwargs `kernel`, `activation`, `N_channels` and no other input arguments. The function must return a Lux layer.  
"""
struct UNet{U} <: Lux.AbstractLuxWrapperLayer{:model}
    model::U
end 

function UNet(; kernel=(3,3), N_channels=[48, 96, 96, 128], in_channels = 3, out_channels=in_channels, activation=NNlib.swish, downconv=ConvBlock, upconv=ConvBlock, upsample= (N_channels::Pair) -> Lux.ConvTranspose((2,2),N_channels[1]=>N_channels[2],stride=2), pool = (N_channels::Pair) -> Lux.Conv((2,2),N_channels[1]=>N_channels[2],stride=2))

    if typeof(N_channels) <: Int 
        N_channels = [N_channels, N_channels, N_channels, N_channels]
    else
        @assert length(N_channels) == 4 "N_channels needs to be an Integer or an Array/Tuple of length 4."
    end

    cat_dim3(a,b) = cat(a,b, dims=3)

    lvl_4 = FlatChain(pool(N_channels[3]=>N_channels[3]), downconv(kernel=kernel, N_channels=N_channels[3]=>N_channels[4], activation=activation), upsample(N_channels[4]=>N_channels[3]))

    lvl_3 = FlatChain(pool(N_channels[2]=>N_channels[2]), downconv(kernel=kernel, N_channels=N_channels[2]=>N_channels[3], activation=activation), 
    Lux.SkipConnection(lvl_4, cat_dim3),  # lvl 4 
    upconv(kernel=kernel, N_channels=(2*N_channels[3])=>N_channels[3], activation=activation), upsample(N_channels[3]=>N_channels[2]))

    lvl_2 = FlatChain(pool(N_channels[1]=>N_channels[1]), downconv(kernel=kernel, N_channels=N_channels[1]=>N_channels[2], activation=activation),
    Lux.SkipConnection(lvl_3, cat_dim3), # lvl 3
    upconv(kernel=kernel, N_channels=(2*N_channels[2])=>N_channels[2], activation=activation), upsample(N_channels[2]=>N_channels[1]))

    return UNet(FlatChain(downconv(kernel=kernel, N_channels=in_channels=>N_channels[1], activation=activation), 
    Lux.SkipConnection(lvl_2, cat_dim3), # lvl 2 
    upconv(kernel=kernel, N_channels=(2*N_channels[1])=>N_channels[1], activation=activation),
    Lux.Conv((1,1), N_channels[1]=>out_channels, identity, pad=Lux.SamePad())))
end 

(m::UNet)(x, ps, st) = m.model(x, ps, st)

outputsize(m::UNet, x::AbstractArray) = (size(x,1),size(x,2),m.model.layers[end].out_chs,size(x,4))
inputsize(m::UNet, x::AbstractArray) =  (size(x,1),size(x,2),m.model.layers[1].in_chs,size(x,4))


# alternative recursive UNet setup without NeuralDE 

