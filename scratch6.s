	.file	"scratch6.4653220bdc1b52a6-cgu.0"
	.section	.text._ZN3std2rt10lang_start17h2fe9acc315706dbfE,"ax",@progbits
	.hidden	_ZN3std2rt10lang_start17h2fe9acc315706dbfE
	.globl	_ZN3std2rt10lang_start17h2fe9acc315706dbfE
	.p2align	4
	.type	_ZN3std2rt10lang_start17h2fe9acc315706dbfE,@function
_ZN3std2rt10lang_start17h2fe9acc315706dbfE:
	.cfi_startproc
	pushq	%rax
	.cfi_def_cfa_offset 16
	movl	%ecx, %r8d
	movq	%rdx, %rcx
	movq	%rsi, %rdx
	movq	%rdi, (%rsp)
	leaq	.Lanon.1d6dbc74351859748c854dca404cbc07.0(%rip), %rsi
	movq	%rsp, %rdi
	callq	*_ZN3std2rt19lang_start_internal17hc68d929ebd5f7eeaE@GOTPCREL(%rip)
	popq	%rcx
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end0:
	.size	_ZN3std2rt10lang_start17h2fe9acc315706dbfE, .Lfunc_end0-_ZN3std2rt10lang_start17h2fe9acc315706dbfE
	.cfi_endproc

	.section	".text._ZN3std2rt10lang_start28_$u7b$$u7b$closure$u7d$$u7d$17hac4228d1dea1f15aE","ax",@progbits
	.p2align	4
	.type	_ZN3std2rt10lang_start28_$u7b$$u7b$closure$u7d$$u7d$17hac4228d1dea1f15aE,@function
_ZN3std2rt10lang_start28_$u7b$$u7b$closure$u7d$$u7d$17hac4228d1dea1f15aE:
	.cfi_startproc
	pushq	%rax
	.cfi_def_cfa_offset 16
	movq	(%rdi), %rdi
	callq	_ZN3std3sys9backtrace28__rust_begin_short_backtrace17h8f8422d954d50b69E
	xorl	%eax, %eax
	popq	%rcx
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end1:
	.size	_ZN3std2rt10lang_start28_$u7b$$u7b$closure$u7d$$u7d$17hac4228d1dea1f15aE, .Lfunc_end1-_ZN3std2rt10lang_start28_$u7b$$u7b$closure$u7d$$u7d$17hac4228d1dea1f15aE
	.cfi_endproc

	.section	.text._ZN3std3sys9backtrace28__rust_begin_short_backtrace17h8f8422d954d50b69E,"ax",@progbits
	.p2align	4
	.type	_ZN3std3sys9backtrace28__rust_begin_short_backtrace17h8f8422d954d50b69E,@function
_ZN3std3sys9backtrace28__rust_begin_short_backtrace17h8f8422d954d50b69E:
	.cfi_startproc
	pushq	%rax
	.cfi_def_cfa_offset 16
	callq	*%rdi
	#APP
	#NO_APP
	popq	%rax
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end2:
	.size	_ZN3std3sys9backtrace28__rust_begin_short_backtrace17h8f8422d954d50b69E, .Lfunc_end2-_ZN3std3sys9backtrace28__rust_begin_short_backtrace17h8f8422d954d50b69E
	.cfi_endproc

	.section	".text._ZN4core3ops8function6FnOnce40call_once$u7b$$u7b$vtable.shim$u7d$$u7d$17ha6ad34eade15db9dE","ax",@progbits
	.p2align	4
	.type	_ZN4core3ops8function6FnOnce40call_once$u7b$$u7b$vtable.shim$u7d$$u7d$17ha6ad34eade15db9dE,@function
_ZN4core3ops8function6FnOnce40call_once$u7b$$u7b$vtable.shim$u7d$$u7d$17ha6ad34eade15db9dE:
	.cfi_startproc
	pushq	%rax
	.cfi_def_cfa_offset 16
	movq	(%rdi), %rdi
	callq	_ZN3std3sys9backtrace28__rust_begin_short_backtrace17h8f8422d954d50b69E
	xorl	%eax, %eax
	popq	%rcx
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end3:
	.size	_ZN4core3ops8function6FnOnce40call_once$u7b$$u7b$vtable.shim$u7d$$u7d$17ha6ad34eade15db9dE, .Lfunc_end3-_ZN4core3ops8function6FnOnce40call_once$u7b$$u7b$vtable.shim$u7d$$u7d$17ha6ad34eade15db9dE
	.cfi_endproc

	.section	.text._ZN8scratch611test_avx51217h2ccfc9e5b9cad23dE,"ax",@progbits
	.p2align	4
	.type	_ZN8scratch611test_avx51217h2ccfc9e5b9cad23dE,@function
_ZN8scratch611test_avx51217h2ccfc9e5b9cad23dE:
	.cfi_startproc
	pushq	%rax
	.cfi_def_cfa_offset 16
	movq	_ZN10std_detect6detect5cache5CACHE17hc42ac9b9518f4abfE@GOTPCREL(%rip), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	je	.LBB4_1
	testl	$524288, %eax
	jne	.LBB4_4
.LBB4_3:
	leaq	.Lanon.1d6dbc74351859748c854dca404cbc07.1(%rip), %rdi
	movl	$21, %esi
	popq	%rax
	.cfi_def_cfa_offset 8
	jmpq	*_ZN3std2io5stdio6_print17hdebbaafb78bfc2d5E@GOTPCREL(%rip)
.LBB4_1:
	.cfi_def_cfa_offset 16
	callq	*_ZN10std_detect6detect5cache21detect_and_initialize17h3b22723e92d00469E@GOTPCREL(%rip)
	testl	$524288, %eax
	je	.LBB4_3
.LBB4_4:
	leaq	.Lanon.1d6dbc74351859748c854dca404cbc07.2(%rip), %rdi
	movl	$23, %esi
	popq	%rax
	.cfi_def_cfa_offset 8
	jmpq	*_ZN3std2io5stdio6_print17hdebbaafb78bfc2d5E@GOTPCREL(%rip)
.Lfunc_end4:
	.size	_ZN8scratch611test_avx51217h2ccfc9e5b9cad23dE, .Lfunc_end4-_ZN8scratch611test_avx51217h2ccfc9e5b9cad23dE
	.cfi_endproc

	.section	.text._ZN8scratch64main17hea5f056acad256e4E,"ax",@progbits
	.hidden	_ZN8scratch64main17hea5f056acad256e4E
	.globl	_ZN8scratch64main17hea5f056acad256e4E
	.p2align	4
	.type	_ZN8scratch64main17hea5f056acad256e4E,@function
_ZN8scratch64main17hea5f056acad256e4E:
	.cfi_startproc
	jmp	_ZN8scratch611test_avx51217h2ccfc9e5b9cad23dE
.Lfunc_end5:
	.size	_ZN8scratch64main17hea5f056acad256e4E, .Lfunc_end5-_ZN8scratch64main17hea5f056acad256e4E
	.cfi_endproc

	.section	.text.main,"ax",@progbits
	.globl	main
	.p2align	4
	.type	main,@function
main:
	.cfi_startproc
	pushq	%rax
	.cfi_def_cfa_offset 16
	movq	%rsi, %rcx
	movslq	%edi, %rdx
	leaq	_ZN8scratch64main17hea5f056acad256e4E(%rip), %rax
	movq	%rax, (%rsp)
	leaq	.Lanon.1d6dbc74351859748c854dca404cbc07.0(%rip), %rsi
	movq	%rsp, %rdi
	xorl	%r8d, %r8d
	callq	*_ZN3std2rt19lang_start_internal17hc68d929ebd5f7eeaE@GOTPCREL(%rip)
	popq	%rcx
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end6:
	.size	main, .Lfunc_end6-main
	.cfi_endproc

	.type	.Lanon.1d6dbc74351859748c854dca404cbc07.0,@object
	.section	.data.rel.ro..Lanon.1d6dbc74351859748c854dca404cbc07.0,"aw",@progbits
	.p2align	3, 0x0
.Lanon.1d6dbc74351859748c854dca404cbc07.0:
	.asciz	"\000\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\b\000\000\000\000\000\000"
	.quad	_ZN4core3ops8function6FnOnce40call_once$u7b$$u7b$vtable.shim$u7d$$u7d$17ha6ad34eade15db9dE
	.quad	_ZN3std2rt10lang_start28_$u7b$$u7b$closure$u7d$$u7d$17hac4228d1dea1f15aE
	.quad	_ZN3std2rt10lang_start28_$u7b$$u7b$closure$u7d$$u7d$17hac4228d1dea1f15aE
	.size	.Lanon.1d6dbc74351859748c854dca404cbc07.0, 48

	.type	.Lanon.1d6dbc74351859748c854dca404cbc07.1,@object
	.section	.rodata..Lanon.1d6dbc74351859748c854dca404cbc07.1,"a",@progbits
.Lanon.1d6dbc74351859748c854dca404cbc07.1:
	.ascii	"No AVX512\n"
	.size	.Lanon.1d6dbc74351859748c854dca404cbc07.1, 10

	.type	.Lanon.1d6dbc74351859748c854dca404cbc07.2,@object
	.section	.rodata..Lanon.1d6dbc74351859748c854dca404cbc07.2,"a",@progbits
.Lanon.1d6dbc74351859748c854dca404cbc07.2:
	.ascii	"Has AVX512\n"
	.size	.Lanon.1d6dbc74351859748c854dca404cbc07.2, 11

	.ident	"rustc version 1.94.1 (e408947bf 2026-03-25)"
	.section	".note.GNU-stack","",@progbits
