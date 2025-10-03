
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf030-for-backward/cf030-for-backward_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000498 <__Z17test_for_backwardv>:
100000498: d10103ff    	sub	sp, sp, #0x40
10000049c: a9037bfd    	stp	x29, x30, [sp, #0x30]
1000004a0: 9100c3fd    	add	x29, sp, #0x30
1000004a4: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
1000004a8: f9400108    	ldr	x8, [x8]
1000004ac: f9400108    	ldr	x8, [x8]
1000004b0: f81f83a8    	stur	x8, [x29, #-0x8]
1000004b4: 90000008    	adrp	x8, 0x100000000 <___stack_chk_guard+0x100000000>
1000004b8: 9115e108    	add	x8, x8, #0x578
1000004bc: 3dc00100    	ldr	q0, [x8]
1000004c0: 3d8007e0    	str	q0, [sp, #0x10]
1000004c4: b9401108    	ldr	w8, [x8, #0x10]
1000004c8: b90023e8    	str	w8, [sp, #0x20]
1000004cc: b9000fff    	str	wzr, [sp, #0xc]
1000004d0: 52800088    	mov	w8, #0x4                ; =4
1000004d4: b9000be8    	str	w8, [sp, #0x8]
1000004d8: 14000001    	b	0x1000004dc <__Z17test_for_backwardv+0x44>
1000004dc: b9400be8    	ldr	w8, [sp, #0x8]
1000004e0: 37f801a8    	tbnz	w8, #0x1f, 0x100000514 <__Z17test_for_backwardv+0x7c>
1000004e4: 14000001    	b	0x1000004e8 <__Z17test_for_backwardv+0x50>
1000004e8: b9800be9    	ldrsw	x9, [sp, #0x8]
1000004ec: 910043e8    	add	x8, sp, #0x10
1000004f0: b8697909    	ldr	w9, [x8, x9, lsl #2]
1000004f4: b9400fe8    	ldr	w8, [sp, #0xc]
1000004f8: 0b090108    	add	w8, w8, w9
1000004fc: b9000fe8    	str	w8, [sp, #0xc]
100000500: 14000001    	b	0x100000504 <__Z17test_for_backwardv+0x6c>
100000504: b9400be8    	ldr	w8, [sp, #0x8]
100000508: 71000508    	subs	w8, w8, #0x1
10000050c: b9000be8    	str	w8, [sp, #0x8]
100000510: 17fffff3    	b	0x1000004dc <__Z17test_for_backwardv+0x44>
100000514: b9400fe8    	ldr	w8, [sp, #0xc]
100000518: b90007e8    	str	w8, [sp, #0x4]
10000051c: f85f83a9    	ldur	x9, [x29, #-0x8]
100000520: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
100000524: f9400108    	ldr	x8, [x8]
100000528: f9400108    	ldr	x8, [x8]
10000052c: eb090108    	subs	x8, x8, x9
100000530: 54000060    	b.eq	0x10000053c <__Z17test_for_backwardv+0xa4>
100000534: 14000001    	b	0x100000538 <__Z17test_for_backwardv+0xa0>
100000538: 9400000d    	bl	0x10000056c <___stack_chk_guard+0x10000056c>
10000053c: b94007e0    	ldr	w0, [sp, #0x4]
100000540: a9437bfd    	ldp	x29, x30, [sp, #0x30]
100000544: 910103ff    	add	sp, sp, #0x40
100000548: d65f03c0    	ret

000000010000054c <_main>:
10000054c: d10083ff    	sub	sp, sp, #0x20
100000550: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000554: 910043fd    	add	x29, sp, #0x10
100000558: b81fc3bf    	stur	wzr, [x29, #-0x4]
10000055c: 97ffffcf    	bl	0x100000498 <__Z17test_for_backwardv>
100000560: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000564: 910083ff    	add	sp, sp, #0x20
100000568: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

000000010000056c <__stubs>:
10000056c: 90000030    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100000570: f9400610    	ldr	x16, [x16, #0x8]
100000574: d61f0200    	br	x16
