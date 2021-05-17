use std::os::windows::prelude::OsStringExt;

use bindings::{
    windows::{self, Abi, Interface, IntoParam},
    Windows::Win32::{
        Graphics::{
            Direct3D11::{ID3DBlob, D3D_FEATURE_LEVEL_12_1, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST},
            Direct3D12::{
                D3D12CreateDevice, D3D12GetDebugInterface, D3D12SerializeRootSignature,
                ID3D12CommandAllocator, ID3D12CommandQueue, ID3D12Debug3, ID3D12DescriptorHeap,
                ID3D12Device, ID3D12Fence, ID3D12GraphicsCommandList, ID3D12PipelineState,
                ID3D12Resource, ID3D12RootSignature, D3D12_BLEND_DESC, D3D12_BLEND_ONE,
                D3D12_BLEND_OP_ADD, D3D12_BLEND_ZERO, D3D12_COLOR_WRITE_ENABLE_ALL,
                D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_DESC,
                D3D12_COMMAND_QUEUE_FLAG_NONE, D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF,
                D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_CULL_MODE_BACK,
                D3D12_DEFAULT_DEPTH_BIAS, D3D12_DEFAULT_DEPTH_BIAS_CLAMP,
                D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS, D3D12_DEPTH_STENCIL_DESC,
                D3D12_DESCRIPTOR_HEAP_DESC, D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                D3D12_DESCRIPTOR_HEAP_TYPE_RTV, D3D12_FENCE_FLAG_NONE, D3D12_FILL_MODE_SOLID,
                D3D12_GRAPHICS_PIPELINE_STATE_DESC, D3D12_HEAP_FLAG_NONE, D3D12_HEAP_PROPERTIES,
                D3D12_HEAP_TYPE_UPLOAD, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                D3D12_INPUT_ELEMENT_DESC, D3D12_INPUT_LAYOUT_DESC, D3D12_LOGIC_OP_NOOP,
                D3D12_MAX_DEPTH, D3D12_MEMORY_POOL_UNKNOWN, D3D12_MIN_DEPTH,
                D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE, D3D12_RANGE, D3D12_RASTERIZER_DESC,
                D3D12_RENDER_TARGET_BLEND_DESC, D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_BARRIER_0,
                D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_BARRIER_FLAG_NONE,
                D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, D3D12_RESOURCE_DESC,
                D3D12_RESOURCE_DIMENSION_BUFFER, D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATE_PRESENT,
                D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_TRANSITION_BARRIER,
                D3D12_ROOT_SIGNATURE_DESC,
                D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
                D3D12_SHADER_BYTECODE, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_VERTEX_BUFFER_VIEW,
                D3D12_VIEWPORT, D3D_ROOT_SIGNATURE_VERSION_1,
            },
            Dxgi::{
                CreateDXGIFactory2, IDXGIAdapter, IDXGIFactory6, IDXGISwapChain3,
                DXGI_ADAPTER_DESC, DXGI_CREATE_FACTORY_DEBUG, DXGI_FORMAT,
                DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_R32G32B32_FLOAT,
                DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_UNKNOWN,
                DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, DXGI_MWA_NO_ALT_ENTER, DXGI_SAMPLE_DESC,
                DXGI_SWAP_CHAIN_DESC1, DXGI_SWAP_EFFECT_FLIP_DISCARD,
                DXGI_USAGE_RENDER_TARGET_OUTPUT,
            },
            Hlsl::{D3DCompile2, D3DCOMPILE_DEBUG, D3DCOMPILE_SKIP_OPTIMIZATION},
        },
        System::{
            Diagnostics::Debug::GetLastError,
            SystemServices::{HANDLE, PSTR},
            Threading::{CreateEventW, WaitForSingleObject},
            WindowsProgramming::{CloseHandle, INFINITE},
        },
        UI::{DisplayDevices::RECT, WindowsAndMessaging::HWND},
    },
};

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    platform::windows::WindowExtWindows,
    window::{Window, WindowBuilder},
};

const FRAME_COUNT: u32 = 2;

struct Vertex((f32, f32, f32), (f32, f32, f32, f32));

fn main() -> windows::Result<()> {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_visible(false)
        .with_title("DirectX 12 Rust example")
        .build(&event_loop)
        .unwrap();
    let window_handle = HWND(window.hwnd() as isize);

    if cfg!(debug_asserts) {
        let debug_interface: ID3D12Debug3 = unsafe { D3D12GetDebugInterface() }?;
        unsafe { debug_interface.EnableDebugLayer() };
        unsafe { debug_interface.SetEnableGPUBasedValidation(true) };
    }

    let factory = create_factory()?;
    let device = create_device(&factory)?;
    let command_queue = create_command_queue(&device)?;
    let swap_chain = create_swap_chain(&window, &factory, &command_queue, window_handle)?;
    let mut frame_index = unsafe { swap_chain.GetCurrentBackBufferIndex() };
    unsafe { factory.MakeWindowAssociation(window_handle, DXGI_MWA_NO_ALT_ENTER) }.ok()?;
    let descriptor_heap = create_descriptor_heap(&device)?;
    let descriptor_heap_size =
        unsafe { device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) };
    let render_targets =
        create_render_targets(&descriptor_heap, &swap_chain, &device, descriptor_heap_size);
    let command_allocator: ID3D12CommandAllocator =
        unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }?;
    let root_signature = create_root_signature(&device)?;
    let shader_source = include_bytes!("shaders/shader.hlsl");
    let vertex_shader = compile_shader(shader_source, "VSMain", "vs_5_0")?;
    let pixel_shader = compile_shader(shader_source, "PSMain", "ps_5_0")?;
    let graphics_pipeline_state =
        create_graphics_pipeline_state(&root_signature, vertex_shader, pixel_shader, &device)?;
    let command_list: ID3D12GraphicsCommandList = unsafe {
        device.CreateCommandList(
            0,
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            &command_allocator,
            &graphics_pipeline_state,
        )
    }?;
    unsafe { command_list.Close() }.ok()?;

    let window_inner_size = window.inner_size();
    let aspect_ratio = window_inner_size.width / window_inner_size.height;

    let vertices = [
        Vertex((0.0, 0.25 * aspect_ratio as f32, 0.0), (1.0, 0.0, 0.0, 1.0)),
        Vertex(
            (0.25, -0.25 * aspect_ratio as f32, 0.0),
            (0.0, 1.0, 0.0, 1.0),
        ),
        Vertex(
            (-0.25, -0.25 * aspect_ratio as f32, 0.0),
            (0.0, 0.0, 1.0, 1.0),
        ),
    ];

    let vertex_buffer: ID3D12Resource = unsafe {
        device.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_UPLOAD,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
                CreationNodeMask: 1,
                VisibleNodeMask: 1,
            },
            D3D12_HEAP_FLAG_NONE,
            &D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Alignment: 0,
                Width: std::mem::size_of_val(&vertices) as u64,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_UNKNOWN,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_NONE,
            },
            D3D12_RESOURCE_STATE_GENERIC_READ,
            std::ptr::null(),
        )
    }?;

    let mut vertex_buffer_data = std::ptr::null_mut();

    unsafe {
        vertex_buffer.Map(
            0,
            &D3D12_RANGE { Begin: 0, End: 0 },
            &mut vertex_buffer_data,
        )
    }
    .ok()?;

    unsafe {
        vertices
            .as_ptr()
            .copy_to_nonoverlapping(vertex_buffer_data.cast(), vertices.len())
    };

    unsafe { vertex_buffer.Unmap(0, std::ptr::null()) };

    let vertex_buffer_view = D3D12_VERTEX_BUFFER_VIEW {
        BufferLocation: unsafe { vertex_buffer.GetGPUVirtualAddress() },
        StrideInBytes: std::mem::size_of::<Vertex>() as u32,
        SizeInBytes: std::mem::size_of_val(&vertices) as u32,
    };

    let mut fence_value = 1;
    let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }?;

    let fence_event = unsafe { CreateEventW(std::ptr::null_mut(), false, false, None) };
    if fence_event.is_null() {
        panic!("Error during event creation: {:?}", unsafe {
            GetLastError()
        });
    }

    let viewport = D3D12_VIEWPORT {
        TopLeftX: 0.0,
        TopLeftY: 0.0,
        Width: window_inner_size.width as f32,
        Height: window_inner_size.height as f32,
        MinDepth: D3D12_MIN_DEPTH,
        MaxDepth: D3D12_MAX_DEPTH,
    };
    let scissort_rect = RECT {
        left: 0,
        top: 0,
        bottom: window_inner_size.height as i32,
        right: window_inner_size.width as i32,
    };

    wait_for_previous_frame(
        &mut fence_value,
        &command_queue,
        &fence,
        &fence_event,
        &mut frame_index,
        &swap_chain,
    );

    window.set_visible(true);
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::WindowEvent {
                window_id: _,
                event,
            } if event == WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
            Event::RedrawRequested(_) => {
                unsafe { command_allocator.Reset() }.unwrap();
                unsafe { command_list.Reset(&command_allocator, &graphics_pipeline_state) }
                    .unwrap();

                unsafe { command_list.SetGraphicsRootSignature(&root_signature) };
                unsafe { command_list.RSSetViewports(1, &viewport) };
                unsafe { command_list.RSSetScissorRects(1, &scissort_rect) };

                unsafe {
                    command_list.ResourceBarrier(
                        1,
                        &D3D12_RESOURCE_BARRIER {
                            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                                Transition: D3D12_RESOURCE_TRANSITION_BARRIER {
                                    pResource: Some(render_targets[frame_index as usize].clone()),
                                    StateBefore: D3D12_RESOURCE_STATE_PRESENT,
                                    StateAfter: D3D12_RESOURCE_STATE_RENDER_TARGET,
                                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                                }
                                .abi(),
                            },
                        },
                    )
                };

                let descriptor_handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: unsafe { descriptor_heap.GetCPUDescriptorHandleForHeapStart() }.ptr
                        + frame_index as usize * descriptor_heap_size as usize,
                };

                unsafe {
                    command_list.OMSetRenderTargets(1, &descriptor_handle, false, std::ptr::null())
                };
                unsafe {
                    command_list.ClearRenderTargetView(
                        descriptor_handle,
                        [0.0, 0.2, 0.4, 1.0].as_ptr(),
                        0,
                        std::ptr::null(),
                    )
                };
                unsafe { command_list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST) };
                unsafe { command_list.IASetVertexBuffers(0, 1, &vertex_buffer_view) };
                unsafe { command_list.DrawInstanced(3, 1, 0, 0) };

                unsafe {
                    command_list.ResourceBarrier(
                        1,
                        &D3D12_RESOURCE_BARRIER {
                            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                                Transition: D3D12_RESOURCE_TRANSITION_BARRIER {
                                    pResource: Some(render_targets[frame_index as usize].clone()),
                                    StateBefore: D3D12_RESOURCE_STATE_RENDER_TARGET,
                                    StateAfter: D3D12_RESOURCE_STATE_PRESENT,
                                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                                }
                                .abi(),
                            },
                        },
                    )
                };

                unsafe { command_list.Close() }.unwrap();

                unsafe {
                    command_queue.ExecuteCommandLists(1, &mut Some(command_list.cast().unwrap()))
                };

                unsafe { swap_chain.Present(1, 0) }.unwrap();

                wait_for_previous_frame(
                    &mut fence_value,
                    &command_queue,
                    &fence,
                    &fence_event,
                    &mut frame_index,
                    &swap_chain,
                );
            }
            Event::LoopDestroyed => {
                wait_for_previous_frame(
                    &mut fence_value,
                    &command_queue,
                    &fence,
                    &fence_event,
                    &mut frame_index,
                    &swap_chain,
                );
                unsafe { CloseHandle(fence_event) };
            }
            _ => {}
        }
    });
}

fn wait_for_previous_frame(
    fence_value: &mut u64,
    command_queue: &ID3D12CommandQueue,
    fence: &ID3D12Fence,
    fence_event: &HANDLE,
    frame_index: &mut u32,
    swap_chain: &IDXGISwapChain3,
) {
    let previous_fence_value = *fence_value;
    unsafe { command_queue.Signal(fence, previous_fence_value).unwrap() };
    *fence_value += 1;
    if unsafe { fence.GetCompletedValue() } < previous_fence_value {
        unsafe { fence.SetEventOnCompletion(previous_fence_value, fence_event) }.unwrap();
        unsafe { WaitForSingleObject(fence_event, INFINITE) };
    }
    *frame_index = unsafe { swap_chain.GetCurrentBackBufferIndex() };
}

fn create_factory() -> windows::Result<IDXGIFactory6> {
    let factory_create_flags = if cfg!(debug_assertions) {
        DXGI_CREATE_FACTORY_DEBUG
    } else {
        0
    };

    unsafe { CreateDXGIFactory2(factory_create_flags) }
}

fn create_device(factory: &IDXGIFactory6) -> windows::Result<ID3D12Device> {
    let adapter: IDXGIAdapter =
        unsafe { factory.EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE) }?;

    if cfg!(debug_assertions) {
        let mut adapter_desc = DXGI_ADAPTER_DESC::default();
        unsafe { adapter.GetDesc(&mut adapter_desc) }.ok()?;
        println!(
            "Using adapter: {}",
            std::ffi::OsString::from_wide(&adapter_desc.Description)
                .to_str()
                .unwrap()
        );
    }

    unsafe { D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_1) }
}

fn create_command_queue(device: &ID3D12Device) -> windows::Result<ID3D12CommandQueue> {
    let command_queue_desc = D3D12_COMMAND_QUEUE_DESC {
        Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
        ..Default::default()
    };

    unsafe { device.CreateCommandQueue(&command_queue_desc) }
}

fn create_swap_chain(
    window: &Window,
    factory: &IDXGIFactory6,
    command_queue: &ID3D12CommandQueue,
    window_handle: HWND,
) -> windows::Result<IDXGISwapChain3> {
    let window_inner_size = window.inner_size();
    let swap_chain_desc = DXGI_SWAP_CHAIN_DESC1 {
        Width: window_inner_size.width,
        Height: window_inner_size.height,
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            ..Default::default()
        },
        BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
        BufferCount: FRAME_COUNT,
        SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
        ..Default::default()
    };

    let mut swap_chain = None;

    let swap_chain = unsafe {
        factory.CreateSwapChainForHwnd(
            command_queue,
            window_handle,
            &swap_chain_desc,
            std::ptr::null(),
            None,
            &mut swap_chain,
        )
    }
    .and_some(swap_chain)?;

    swap_chain.cast()
}

fn create_descriptor_heap(device: &ID3D12Device) -> windows::Result<ID3D12DescriptorHeap> {
    let descriptor_heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
        NumDescriptors: FRAME_COUNT,
        Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
        ..Default::default()
    };

    unsafe { device.CreateDescriptorHeap(&descriptor_heap_desc) }
}

fn create_render_targets(
    descriptor_heap: &ID3D12DescriptorHeap,
    swap_chain: &IDXGISwapChain3,
    device: &ID3D12Device,
    descriptor_heap_size: u32,
) -> Vec<ID3D12Resource> {
    let mut descriptor_handle = unsafe { descriptor_heap.GetCPUDescriptorHandleForHeapStart() };

    (0..FRAME_COUNT)
        .filter_map(|i| {
            let render_target = unsafe { swap_chain.GetBuffer(i) }.ok()?;

            unsafe {
                device.CreateRenderTargetView(&render_target, std::ptr::null(), descriptor_handle)
            };

            descriptor_handle.ptr += descriptor_heap_size as usize;

            Some(render_target)
        })
        .collect()
}

fn create_root_signature(device: &ID3D12Device) -> windows::Result<ID3D12RootSignature> {
    let root_signature_desc = D3D12_ROOT_SIGNATURE_DESC {
        Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
        ..Default::default()
    };
    let mut signature = None;
    let mut error = None;
    let signature = unsafe {
        D3D12SerializeRootSignature(
            &root_signature_desc,
            D3D_ROOT_SIGNATURE_VERSION_1,
            &mut signature,
            &mut error,
        )
    }
    .and_some(signature)?;

    unsafe {
        device.CreateRootSignature(0, signature.GetBufferPointer(), signature.GetBufferSize())
    }
}
fn compile_shader<'a>(
    shader_source: &[u8],
    shader_entry_point: impl IntoParam<'a, PSTR>,
    shader_target_name: impl IntoParam<'a, PSTR>,
) -> windows::Result<ID3DBlob> {
    let compiler_flags = if cfg!(debug_assertions) {
        D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION
    } else {
        0
    };

    let mut shader = None;
    let mut shader_error = None;

    let shader = unsafe {
        D3DCompile2(
            shader_source.as_ptr().cast(),
            shader_source.len(),
            None,
            std::ptr::null(),
            None,
            shader_entry_point,
            shader_target_name,
            compiler_flags,
            0,
            0,
            std::ptr::null(),
            0,
            &mut shader,
            &mut shader_error,
        )
    }
    .and_some(shader);

    if let Some(shader_error) = shader_error {
        eprintln!(
            "Error during shader compilation: {}",
            unsafe {
                std::str::from_utf8(std::slice::from_raw_parts(
                    shader_error.GetBufferPointer() as *const u8,
                    shader_error.GetBufferSize(),
                ))
            }
            .unwrap()
        )
    }

    shader
}

fn create_graphics_pipeline_state(
    root_signature: &ID3D12RootSignature,
    vertex_shader: ID3DBlob,
    pixel_shader: ID3DBlob,
    device: &ID3D12Device,
) -> windows::Result<ID3D12PipelineState> {
    let mut input_element_descs = [
        D3D12_INPUT_ELEMENT_DESC {
            SemanticName: PSTR(b"POSITION\0".as_ptr() as _),
            SemanticIndex: 0,
            Format: DXGI_FORMAT_R32G32B32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: 0,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
        D3D12_INPUT_ELEMENT_DESC {
            SemanticName: PSTR(b"COLOR\0".as_ptr() as _),
            SemanticIndex: 0,
            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: 12,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
    ];

    let graphics_pipeline_state_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        InputLayout: D3D12_INPUT_LAYOUT_DESC {
            pInputElementDescs: input_element_descs.as_mut_ptr(),
            NumElements: input_element_descs.len() as u32,
        },
        pRootSignature: Some(root_signature.clone()),
        VS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: unsafe { vertex_shader.GetBufferPointer() },
            BytecodeLength: unsafe { vertex_shader.GetBufferSize() },
        },
        PS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: unsafe { pixel_shader.GetBufferPointer() },
            BytecodeLength: unsafe { pixel_shader.GetBufferSize() },
        },
        RasterizerState: D3D12_RASTERIZER_DESC {
            FillMode: D3D12_FILL_MODE_SOLID,
            CullMode: D3D12_CULL_MODE_BACK,
            FrontCounterClockwise: false.into(),
            DepthBias: D3D12_DEFAULT_DEPTH_BIAS as i32,
            DepthBiasClamp: D3D12_DEFAULT_DEPTH_BIAS_CLAMP,
            SlopeScaledDepthBias: D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS,
            DepthClipEnable: true.into(),
            MultisampleEnable: false.into(),
            AntialiasedLineEnable: false.into(),
            ForcedSampleCount: 0,
            ConservativeRaster: D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF,
        },
        BlendState: D3D12_BLEND_DESC {
            AlphaToCoverageEnable: false.into(),
            IndependentBlendEnable: false.into(),
            RenderTarget: [D3D12_RENDER_TARGET_BLEND_DESC {
                BlendEnable: false.into(),
                LogicOpEnable: false.into(),
                SrcBlend: D3D12_BLEND_ONE,
                DestBlend: D3D12_BLEND_ZERO,
                BlendOp: D3D12_BLEND_OP_ADD,
                SrcBlendAlpha: D3D12_BLEND_ONE,
                DestBlendAlpha: D3D12_BLEND_ZERO,
                BlendOpAlpha: D3D12_BLEND_OP_ADD,
                LogicOp: D3D12_LOGIC_OP_NOOP,
                RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
            }; 8],
        },
        DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
            DepthEnable: false.into(),
            StencilEnable: false.into(),
            ..Default::default()
        },
        SampleMask: u32::MAX,
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: 1,
        RTVFormats: {
            let mut rtv_formats = [DXGI_FORMAT::default(); 8];
            rtv_formats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
            rtv_formats
        },
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            ..Default::default()
        },
        ..Default::default()
    };

    unsafe { device.CreateGraphicsPipelineState(&graphics_pipeline_state_desc) }
}
