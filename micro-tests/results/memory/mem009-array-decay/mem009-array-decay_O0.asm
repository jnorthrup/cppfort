
/Users/jim/work/cppfort/micro-tests/results/memory/mem009-array-decay/mem009-array-decay_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000498 <__Z16test_array_decayPi>:
100000498: d10043ff    	sub	sp, sp, #0x10
10000049c: f90007e0    	str	x0, [sp, #0x8]
1000004a0: f94007e8    	ldr	x8, [sp, #0x8]
1000004a4: b9400100    	ldr	w0, [x8]
1000004a8: 910043ff    	add	sp, sp, #0x10
1000004ac: d65f03c0    	ret

00000001000004b0 <_main>:
1000004b0: d100c3ff    	sub	sp, sp, #0x30
1000004b4: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000004b8: 910083fd    	add	x29, sp, #0x20
1000004bc: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
1000004c0: f9400108    	ldr	x8, [x8]
1000004c4: f9400108    	ldr	x8, [x8]
1000004c8: f81f83a8    	stur	x8, [x29, #-0x8]
1000004cc: b90007ff    	str	wzr, [sp, #0x4]
1000004d0: 90000008    	adrp	x8, 0x100000000 <___stack_chk_guard+0x100000000>
1000004d4: 9114c108    	add	x8, x8, #0x530
1000004d8: f9400109    	ldr	x9, [x8]
1000004dc: 910023e0    	add	x0, sp, #0x8
1000004e0: f90007e9    	str	x9, [sp, #0x8]
1000004e4: b9400908    	ldr	w8, [x8, #0x8]
1000004e8: b90013e8    	str	w8, [sp, #0x10]
1000004ec: 97ffffeb    	bl	0x100000498 <__Z16test_array_decayPi>
1000004f0: b90003e0    	str	w0, [sp]
1000004f4: f85f83a9    	ldur	x9, [x29, #-0x8]
1000004f8: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
1000004fc: f9400108    	ldr	x8, [x8]
100000500: f9400108    	ldr	x8, [x8]
100000504: eb090108    	subs	x8, x8, x9
100000508: 54000060    	b.eq	0x100000514 <_main+0x64>
10000050c: 14000001    	b	0x100000510 <_main+0x60>
100000510: 94000005    	bl	0x100000524 <___stack_chk_guard+0x100000524>
100000514: b94003e0    	ldr	w0, [sp]
100000518: a9427bfd    	ldp	x29, x30, [sp, #0x20]
10000051c: 9100c3ff    	add	sp, sp, #0x30
100000520: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

0000000100000524 <__stubs>:
100000524: 90000030    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100000528: f9400610    	ldr	x16, [x16, #0x8]
10000052c: d61f0200    	br	x16
