
/Users/jim/work/cppfort/micro-tests/results/memory/mem090-array-sizeof/mem090-array-sizeof_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000498 <__Z17test_array_sizeofv>:
100000498: d100c3ff    	sub	sp, sp, #0x30
10000049c: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000004a0: 910083fd    	add	x29, sp, #0x20
1000004a4: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
1000004a8: f9400108    	ldr	x8, [x8]
1000004ac: f9400108    	ldr	x8, [x8]
1000004b0: f81f83a8    	stur	x8, [x29, #-0x8]
1000004b4: 90000008    	adrp	x8, 0x100000000 <___stack_chk_guard+0x100000000>
1000004b8: 9114a108    	add	x8, x8, #0x528
1000004bc: 3dc00100    	ldr	q0, [x8]
1000004c0: 3d8003e0    	str	q0, [sp]
1000004c4: b9401108    	ldr	w8, [x8, #0x10]
1000004c8: b90013e8    	str	w8, [sp, #0x10]
1000004cc: f85f83a9    	ldur	x9, [x29, #-0x8]
1000004d0: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
1000004d4: f9400108    	ldr	x8, [x8]
1000004d8: f9400108    	ldr	x8, [x8]
1000004dc: eb090108    	subs	x8, x8, x9
1000004e0: 54000060    	b.eq	0x1000004ec <__Z17test_array_sizeofv+0x54>
1000004e4: 14000001    	b	0x1000004e8 <__Z17test_array_sizeofv+0x50>
1000004e8: 9400000d    	bl	0x10000051c <___stack_chk_guard+0x10000051c>
1000004ec: 528000a0    	mov	w0, #0x5                ; =5
1000004f0: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000004f4: 9100c3ff    	add	sp, sp, #0x30
1000004f8: d65f03c0    	ret

00000001000004fc <_main>:
1000004fc: d10083ff    	sub	sp, sp, #0x20
100000500: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000504: 910043fd    	add	x29, sp, #0x10
100000508: b81fc3bf    	stur	wzr, [x29, #-0x4]
10000050c: 97ffffe3    	bl	0x100000498 <__Z17test_array_sizeofv>
100000510: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000514: 910083ff    	add	sp, sp, #0x20
100000518: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

000000010000051c <__stubs>:
10000051c: 90000030    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100000520: f9400610    	ldr	x16, [x16, #0x8]
100000524: d61f0200    	br	x16
