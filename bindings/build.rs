fn main() {
    windows::build!(
        Windows::Win32::Debug::GetLastError,
        Windows::Win32::Direct3D11::{ID3DBlob, D3D_FEATURE_LEVEL, D3D_PRIMITIVE_TOPOLOGY},
        Windows::Win32::Direct3D12::*,
        Windows::Win32::Direct3DHlsl::{D3DCompile2, D3DCOMPILE_DEBUG, D3DCOMPILE_SKIP_OPTIMIZATION},
        Windows::Win32::DisplayDevices::RECT,
        Windows::Win32::Dxgi::*,
        Windows::Win32::SystemServices::{CreateEventW, WaitForSingleObject, HANDLE, PSTR},
        Windows::Win32::WindowsAndMessaging::HWND,
        Windows::Win32::WindowsProgramming::{CloseHandle, INFINITE},
    )
}
