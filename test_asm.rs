pub fn main() {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let mut mxcsr: u32 = 0;
        core::arch::asm!(
            "stmxcsr [{0}]",
            in(reg) &mut mxcsr
        );
        mxcsr |= (1 << 15) | (1 << 6);
        core::arch::asm!(
            "ldmxcsr [{0}]",
            in(reg) &mxcsr
        );
        println!("mxcsr is {:x}", mxcsr);
    }
}
