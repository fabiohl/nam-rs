use crate::math::common::{AlignedVec, SimdMath};

/// Camada Densa 1x1 (O Mixador de Canais):
/// Pense nesta camada como uma 'mesa de som digital'. Ela mistura os diversos
/// canais de áudio vindos da etapa anterior para criar a combinação final de timbres.
#[derive(Clone)]
pub struct DenseLayer<const IN: usize, const OUT: usize> {
    /// Matriz de pesos: Define 'quanto' de cada canal entra na mistura.
    pub weights: AlignedVec<u16>,
    /// Bias: Um ajuste de 'volume' básico para cada canal de saída.
    pub bias: AlignedVec<f32>,
    /// Flag que indica se o bias deve ser aplicado.
    pub do_bias: bool,
}

impl<const IN: usize, const OUT: usize> DenseLayer<IN, OUT> {
    /// Processamento Fundido (Fused):
    /// Multiplica, aplica o bias e soma ao resultado, tudo em um único passo matemático.
    /// É a forma mais eficiente de processar um único frame de áudio.
    ///
    /// # Safety
    /// O chamador deve garantir que `in_frame` e `out_frame` tenham tamanhos compatíveis com `IN` e `OUT`.
    #[inline(always)]
    pub unsafe fn process_fused<M: SimdMath>(&self, in_frame: &[f32], out_frame: &mut [f32]) {
        unsafe {
            M::fused_add_gemv(in_frame, &self.weights, &self.bias, out_frame, self.do_bias);
        }
    }

    /// Alias para `process_fused`.
    ///
    /// # Safety
    /// Depende da validade dos buffers e da trait `SimdMath`.
    #[inline(always)]
    pub unsafe fn process_acc_single_frame<M: SimdMath>(
        &self,
        in_frame: &[f32],
        out_frame: &mut [f32],
    ) {
        unsafe { self.process_fused::<M>(in_frame, out_frame) }
    }

    /// Processamento 'Limpo' (Overwrite):
    /// Similar ao fundido, mas substitui o que estiver no buffer de saída
    /// em vez de somar ao valor existente.
    ///
    /// # Safety
    /// O chamador deve garantir que `in_frame` e `out_frame` tenham tamanhos compatíveis com `IN` e `OUT`.
    #[inline(always)]
    pub unsafe fn process_single_frame<M: SimdMath>(
        &self,
        in_frame: &[f32],
        out_frame: &mut [f32],
    ) {
        unsafe {
            M::gemv_overwrite(in_frame, &self.weights, &self.bias, out_frame, self.do_bias);
        }
    }

    /// Processamento de Bloco Fundido:
    /// Versão otimizada para processar várias amostras (batches) de uma só vez,
    /// ganhando muita velocidade em processadores modernos.
    ///
    /// # Safety
    /// O chamador deve garantir que `input` e `output` tenham tamanhos compatíveis com `IN` e `OUT` e `num_frames`.
    #[inline(always)]
    pub unsafe fn process_fused_block<M: SimdMath>(
        &self,
        input: &[f32],
        output: &mut [f32],
        num_frames: usize,
    ) {
        unsafe {
            M::fused_add_gemm_batch(
                input,
                &self.weights,
                &self.bias,
                output,
                num_frames,
                self.do_bias,
            );
        }
    }

    /// Alias para `process_fused_block`.
    ///
    /// # Safety
    /// Depende da validade dos buffers e da trait `SimdMath`.
    #[inline(always)]
    pub unsafe fn process_acc_block<M: SimdMath>(
        &self,
        input: &[f32],
        output: &mut [f32],
        num_frames: usize,
    ) {
        unsafe { self.process_fused_block::<M>(input, output, num_frames) }
    }

    /// Soma do Residual (O 'Atalho' Final):
    /// Esta função faz algo incrível: ela mistura os canais E soma o som original
    /// (residual) ao resultado, tudo sem precisar copiar dados extras na memória.
    ///
    /// # Safety
    /// O chamador deve garantir tamanhos compatíveis e validade dos buffers.
    #[inline(always)]
    pub unsafe fn process_residual_batch<M: SimdMath>(
        &self,
        input: &[f32],
        residual: &[f32],
        output: &mut [f32],
        num_frames: usize,
    ) {
        unsafe {
            M::fused_gemm_residual_batch(
                input,
                &self.weights,
                &self.bias,
                residual,
                output,
                num_frames,
                self.do_bias,
            );
        }
    }

    #[inline(always)]
    /// Processa bloco iterativo substituindo (OVERWRITE) os valores passados em vez de acumular.
    ///
    /// # Safety
    /// O chamador deve garantir que `input` e `output` tenham tamanhos compatíveis com `IN` e `OUT` e `num_frames`.
    pub unsafe fn process_block<M: SimdMath>(
        &self,
        input: &[f32],
        output: &mut [f32],
        num_frames: usize,
    ) {
        unsafe {
            M::gemv_overwrite_batch(
                input,
                &self.weights,
                &self.bias,
                output,
                num_frames,
                self.do_bias,
            );
        }
    }

    ///
    /// # Safety
    /// O chamador deve garantir que `input` e `output` tenham tamanhos
    /// compatíveis com as dimensões `IN` e `OUT` da camada, e que as instruções
    /// SIMD solicitadas pelo despachante `M` estejam disponíveis.
    pub unsafe fn process_bf16<M: SimdMath>(&self, input: &[u16], output: &mut [f32]) {
        let num_frames = output.len() / OUT;
        unsafe {
            M::gemv_overwrite_batch_bf16(
                input,
                &self.weights,
                &self.bias,
                output,
                num_frames,
                self.do_bias,
            );
        }
    }
}
