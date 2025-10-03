
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf088-short-circuit-array-access/cf088-short-circuit-array-access_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000498 <__Z24test_array_short_circuitPii>:
100000498: d10083ff    	sub	sp, sp, #0x20
10000049c: f9000fe0    	str	x0, [sp, #0x18]
1000004a0: b90017e1    	str	w1, [sp, #0x14]
1000004a4: b90013ff    	str	wzr, [sp, #0x10]
1000004a8: b9000fff    	str	wzr, [sp, #0xc]
1000004ac: 14000001    	b	0x1000004b0 <__Z24test_array_short_circuitPii+0x18>
1000004b0: b9400fe9    	ldr	w9, [sp, #0xc]
1000004b4: b94017ea    	ldr	w10, [sp, #0x14]
1000004b8: 52800008    	mov	w8, #0x0                ; =0
1000004bc: 6b0a0129    	subs	w9, w9, w10
1000004c0: b9000be8    	str	w8, [sp, #0x8]
1000004c4: 540001ca    	b.ge	0x1000004fc <__Z24test_array_short_circuitPii+0x64>
1000004c8: 14000001    	b	0x1000004cc <__Z24test_array_short_circuitPii+0x34>
1000004cc: f9400fe8    	ldr	x8, [sp, #0x18]
1000004d0: 52800009    	mov	w9, #0x0                ; =0
1000004d4: b9000be9    	str	w9, [sp, #0x8]
1000004d8: b4000128    	cbz	x8, 0x1000004fc <__Z24test_array_short_circuitPii+0x64>
1000004dc: 14000001    	b	0x1000004e0 <__Z24test_array_short_circuitPii+0x48>
1000004e0: f9400fe8    	ldr	x8, [sp, #0x18]
1000004e4: b9800fe9    	ldrsw	x9, [sp, #0xc]
1000004e8: b8697908    	ldr	w8, [x8, x9, lsl #2]
1000004ec: 71000108    	subs	w8, w8, #0x0
1000004f0: 1a9f07e8    	cset	w8, ne
1000004f4: b9000be8    	str	w8, [sp, #0x8]
1000004f8: 14000001    	b	0x1000004fc <__Z24test_array_short_circuitPii+0x64>
1000004fc: b9400be8    	ldr	w8, [sp, #0x8]
100000500: 360001a8    	tbz	w8, #0x0, 0x100000534 <__Z24test_array_short_circuitPii+0x9c>
100000504: 14000001    	b	0x100000508 <__Z24test_array_short_circuitPii+0x70>
100000508: f9400fe8    	ldr	x8, [sp, #0x18]
10000050c: b9800fe9    	ldrsw	x9, [sp, #0xc]
100000510: b8697909    	ldr	w9, [x8, x9, lsl #2]
100000514: b94013e8    	ldr	w8, [sp, #0x10]
100000518: 0b090108    	add	w8, w8, w9
10000051c: b90013e8    	str	w8, [sp, #0x10]
100000520: 14000001    	b	0x100000524 <__Z24test_array_short_circuitPii+0x8c>
100000524: b9400fe8    	ldr	w8, [sp, #0xc]
100000528: 11000508    	add	w8, w8, #0x1
10000052c: b9000fe8    	str	w8, [sp, #0xc]
100000530: 17ffffe0    	b	0x1000004b0 <__Z24test_array_short_circuitPii+0x18>
100000534: b94013e0    	ldr	w0, [sp, #0x10]
100000538: 910083ff    	add	sp, sp, #0x20
10000053c: d65f03c0    	ret

0000000100000540 <_main>:
100000540: d10103ff    	sub	sp, sp, #0x40
100000544: a9037bfd    	stp	x29, x30, [sp, #0x30]
100000548: 9100c3fd    	add	x29, sp, #0x30
10000054c: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
100000550: f9400108    	ldr	x8, [x8]
100000554: f9400108    	ldr	x8, [x8]
100000558: f81f83a8    	stur	x8, [x29, #-0x8]
10000055c: b9000fff    	str	wzr, [sp, #0xc]
100000560: 90000008    	adrp	x8, 0x100000000 <___stack_chk_guard+0x100000000>
100000564: 91171108    	add	x8, x8, #0x5c4
100000568: 3dc00100    	ldr	q0, [x8]
10000056c: 910043e0    	add	x0, sp, #0x10
100000570: 3d8007e0    	str	q0, [sp, #0x10]
100000574: b9401108    	ldr	w8, [x8, #0x10]
100000578: b90023e8    	str	w8, [sp, #0x20]
10000057c: 528000a1    	mov	w1, #0x5                ; =5
100000580: 97ffffc6    	bl	0x100000498 <__Z24test_array_short_circuitPii>
100000584: b9000be0    	str	w0, [sp, #0x8]
100000588: f85f83a9    	ldur	x9, [x29, #-0x8]
10000058c: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
100000590: f9400108    	ldr	x8, [x8]
100000594: f9400108    	ldr	x8, [x8]
100000598: eb090108    	subs	x8, x8, x9
10000059c: 54000060    	b.eq	0x1000005a8 <_main+0x68>
1000005a0: 14000001    	b	0x1000005a4 <_main+0x64>
1000005a4: 94000005    	bl	0x1000005b8 <___stack_chk_guard+0x1000005b8>
1000005a8: b9400be0    	ldr	w0, [sp, #0x8]
1000005ac: a9437bfd    	ldp	x29, x30, [sp, #0x30]
1000005b0: 910103ff    	add	sp, sp, #0x40
1000005b4: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

00000001000005b8 <__stubs>:
1000005b8: 90000030    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000005bc: f9400610    	ldr	x16, [x16, #0x8]
1000005c0: d61f0200    	br	x16
