
/Users/jim/work/cppfort/micro-tests/results/memory/mem010-vla/mem010-vla_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000448 <__Z8test_vlai>:
100000448: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
10000044c: 910003fd    	mov	x29, sp
100000450: d10043ff    	sub	sp, sp, #0x10
100000454: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
100000458: f9400908    	ldr	x8, [x8, #0x10]
10000045c: f9400108    	ldr	x8, [x8]
100000460: f81f83a8    	stur	x8, [x29, #-0x8]
100000464: d37e7c09    	ubfiz	x9, x0, #2, #32
100000468: 91003d28    	add	x8, x9, #0xf
10000046c: 927c7908    	and	x8, x8, #0x7fffffff0
100000470: 90000030    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100000474: f9400210    	ldr	x16, [x16]
100000478: d63f0200    	blr	x16
10000047c: 910003e9    	mov	x9, sp
100000480: cb080128    	sub	x8, x9, x8
100000484: 9100011f    	mov	sp, x8
100000488: 7100041f    	cmp	w0, #0x1
10000048c: 540000eb    	b.lt	0x1000004a8 <__Z8test_vlai+0x60>
100000490: d2800009    	mov	x9, #0x0                ; =0
100000494: 2a0003ea    	mov	w10, w0
100000498: b8297909    	str	w9, [x8, x9, lsl #2]
10000049c: 91000529    	add	x9, x9, #0x1
1000004a0: eb09015f    	cmp	x10, x9
1000004a4: 54ffffa1    	b.ne	0x100000498 <__Z8test_vlai+0x50>
1000004a8: 8b20c908    	add	x8, x8, w0, sxtw #2
1000004ac: b85fc100    	ldur	w0, [x8, #-0x4]
1000004b0: f85f83a8    	ldur	x8, [x29, #-0x8]
1000004b4: 90000029    	adrp	x9, 0x100004000 <___stack_chk_guard+0x100004000>
1000004b8: f9400929    	ldr	x9, [x9, #0x10]
1000004bc: f9400129    	ldr	x9, [x9]
1000004c0: eb08013f    	cmp	x9, x8
1000004c4: 54000081    	b.ne	0x1000004d4 <__Z8test_vlai+0x8c>
1000004c8: 910003bf    	mov	sp, x29
1000004cc: a8c17bfd    	ldp	x29, x30, [sp], #0x10
1000004d0: d65f03c0    	ret
1000004d4: 94000019    	bl	0x100000538 <___stack_chk_guard+0x100000538>

00000001000004d8 <_main>:
1000004d8: d100c3ff    	sub	sp, sp, #0x30
1000004dc: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000004e0: 910083fd    	add	x29, sp, #0x20
1000004e4: d2800008    	mov	x8, #0x0                ; =0
1000004e8: 90000029    	adrp	x9, 0x100004000 <___stack_chk_guard+0x100004000>
1000004ec: f9400929    	ldr	x9, [x9, #0x10]
1000004f0: f9400129    	ldr	x9, [x9]
1000004f4: f81f83a9    	stur	x9, [x29, #-0x8]
1000004f8: 910013e9    	add	x9, sp, #0x4
1000004fc: b8287928    	str	w8, [x9, x8, lsl #2]
100000500: 91000508    	add	x8, x8, #0x1
100000504: f100151f    	cmp	x8, #0x5
100000508: 54ffffa1    	b.ne	0x1000004fc <_main+0x24>
10000050c: b94017e0    	ldr	w0, [sp, #0x14]
100000510: f85f83a8    	ldur	x8, [x29, #-0x8]
100000514: 90000029    	adrp	x9, 0x100004000 <___stack_chk_guard+0x100004000>
100000518: f9400929    	ldr	x9, [x9, #0x10]
10000051c: f9400129    	ldr	x9, [x9]
100000520: eb08013f    	cmp	x9, x8
100000524: 54000081    	b.ne	0x100000534 <_main+0x5c>
100000528: a9427bfd    	ldp	x29, x30, [sp, #0x20]
10000052c: 9100c3ff    	add	sp, sp, #0x30
100000530: d65f03c0    	ret
100000534: 94000001    	bl	0x100000538 <___stack_chk_guard+0x100000538>

Disassembly of section __TEXT,__stubs:

0000000100000538 <__stubs>:
100000538: 90000030    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
10000053c: f9400610    	ldr	x16, [x16, #0x8]
100000540: d61f0200    	br	x16
