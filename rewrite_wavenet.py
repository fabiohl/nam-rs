import re

with open("src/models/wavenet.rs", "r") as f:
    text = f.read()

# 1. Imports
text = re.sub(
    r"use crate::math::simd::dot_product_avx2;\n\n#\[cfg\(target_arch = \"x86_64\"\)]\nuse crate::math::fastmath::simd_tanh;\n#\[cfg\(target_arch = \"x86_64\"\)]\nuse core::arch::x86_64::\*;\n",
    "use crate::math::simd::SimdMathConfig;\n",
    text
)

# 2. process_frame
text = re.sub(
    r"#\[cfg\(target_arch = \"x86_64\"\)]\n\s*#\[target_feature\(enable = \"avx2,fma\"\)]\n\s*pub unsafe fn process_frame\(\n\s*&self,\n\s*layer_buffer: &\[f32],\n\s*block: &mut \[f32],\n\s*buffer_start: usize,\n\s*\) {",
    "#[cfg(target_arch = \"x86_64\")]\n    pub unsafe fn process_frame(\n        &self,\n        layer_buffer: &[f32],\n        block: &mut [f32],\n        buffer_start: usize,\n        math: &SimdMathConfig,\n    ) {",
    text
)
text = text.replace("sum += dot_product_avx2(in_slice, weight_slice);", "sum += (math.dot_product)(in_slice, weight_slice);")

# 3. DenseLayer process_acc
text = re.sub(
    r"#\[cfg\(target_arch = \"x86_64\"\)]\n\s*#\[target_feature\(enable = \"avx2,fma\"\)]\n\s*pub unsafe fn process_acc\(&self, input: &\[f32], output: &mut \[f32]\) {",
    "#[cfg(target_arch = \"x86_64\")]\n    pub unsafe fn process_acc(&self, input: &[f32], output: &mut [f32], math: &SimdMathConfig) {",
    text
)

text = re.sub(
    r"#\[cfg\(target_arch = \"x86_64\"\)]\n\s*#\[target_feature\(enable = \"avx2,fma\"\)]\n\s*pub unsafe fn process\(&self, input: &\[f32], output: &mut \[f32]\) {",
    "#[cfg(target_arch = \"x86_64\")]\n    pub unsafe fn process(&self, input: &[f32], output: &mut [f32], math: &SimdMathConfig) {",
    text
)

text = re.sub(r"let sum = unsafe \{ dot_product_avx2\(input, weight_slice\) \};", "let sum = unsafe { (math.dot_product)(input, weight_slice) };", text)

# 4. WaveNetLayer process
text = re.sub(
    r"#\[cfg\(target_arch = \"x86_64\"\)]\n\s*#\[target_feature\(enable = \"avx2,fma\"\)]\n\s*pub unsafe fn process\(\n\s*&self,\n\s*condition: &\[f32],\n\s*head_input: &mut \[f32],\n\s*output: &mut \[f32],\n\s*layer_buffer: &\[f32],\n\s*buffer_start: usize,\n\s*\) {",
    "#[cfg(target_arch = \"x86_64\")]\n    pub unsafe fn process(\n        &self,\n        condition: &[f32],\n        head_input: &mut [f32],\n        output: &mut [f32],\n        layer_buffer: &[f32],\n        buffer_start: usize,\n        math: &SimdMathConfig,\n    ) {",
    text
)
text = text.replace("self.conv1d\n                .process_frame(layer_buffer, &mut block, buffer_start);", "self.conv1d\n                .process_frame(layer_buffer, &mut block, buffer_start, math);")
text = text.replace("self.input_mixin.process_acc(condition, &mut block);", "self.input_mixin.process_acc(condition, &mut block, math);")
text = text.replace("self.one_by_one.process(&block, output);", "self.one_by_one.process(&block, output, math);")

tanh_section = """            // Ativação Tanh usando Intrínsecos Vetorizados
            let mut i = 0;
            while i + 8 <= CH {
                let va = _mm256_loadu_ps(block.as_ptr().add(i));
                let vt = simd_tanh(va);
                _mm256_storeu_ps(block.as_mut_ptr().add(i), vt);
                i += 8;
            }
            while i < CH {
                block[i] = block[i].tanh();
                i += 1;
            }"""
text = text.replace(tanh_section, "            // Ativação Tanh usando V-Table do SIMD HW Config\n            (math.tanh_slice)(&mut block);")


# 5. Array process & prewarm
text = re.sub(
    r"#\[cfg\(target_arch = \"x86_64\"\)]\n\s*pub unsafe fn process\(&mut self, layer_inputs: &\[f32], condition: &\[f32]\) {",
    "#[cfg(target_arch = \"x86_64\")]\n    pub unsafe fn process(&mut self, layer_inputs: &[f32], condition: &[f32], math: &SimdMathConfig) {",
    text
)
text = text.replace(".process(layer_inputs, &mut state_0.layer_buffer[start..start + CH]);", ".process(layer_inputs, &mut state_0.layer_buffer[start..start + CH], math);")
text = text.replace("current_state.buffer_start,\n                    );", "current_state.buffer_start,\n                        math,\n                    );")
text = text.replace(".process(&self.head_accum[0..CH], &mut self.head_outputs[0..HEAD]);", ".process(&self.head_accum[0..CH], &mut self.head_outputs[0..HEAD], math);")


text = text.replace("pub fn prewarm(&mut self, layer_inputs: &[f32], condition: &[f32]) {", "pub fn prewarm(&mut self, layer_inputs: &[f32], condition: &[f32], math: &SimdMathConfig) {")

# 6. Model process & prewarm
text = text.replace("self.array1.process(&layer_inputs_1, &condition);", "self.array1.process(&layer_inputs_1, &condition, math);")
text = text.replace("self.array2.process(array1_outputs, &condition);", "self.array2.process(array1_outputs, &condition, math);")

text = text.replace("pub fn process(&mut self, input: &[f32], output: &mut [f32]) {", "pub fn process(&mut self, input: &[f32], output: &mut [f32]) {\n        let math = &crate::math::simd::SimdMathConfig::current();")
text = text.replace("pub fn prewarm(&mut self) {", "pub fn prewarm(&mut self) {\n        let math = &crate::math::simd::SimdMathConfig::current();")

text = text.replace("self.array1.prewarm(&layer_inputs_1, &condition);", "self.array1.prewarm(&layer_inputs_1, &condition, math);")
text = text.replace("self.array2.prewarm(array1_outputs, &condition);", "self.array2.prewarm(array1_outputs, &condition, math);")


# Remove safety notice about FMA where we removed target feature
text = text.replace("Depende nativamente do conjunto de instruções `AVX2` e `FMA`.", "Depende dinamicamente da V-Table `SimdMathConfig` fornecida.")
text = text.replace("Requer suporte dinâmico a AVX2 e FMA no Host.", "Despacho matemático via ponteiro para funções intrínsecas inlined.")


# Test updates
test_updates = """    fn build_tiny_wavenet() -> WaveNetModel<4, 3, 2> {
        let math = crate::math::simd::SimdMathConfig::current();"""

text = text.replace("    fn build_tiny_wavenet() -> WaveNetModel<4, 3, 2> {", test_updates)

with open("src/models/wavenet.rs", "w") as f:
    f.write(text)

