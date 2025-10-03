
/Users/jim/work/cppfort/micro-tests/results/memory/mem100-array-max/mem100-array-max_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000498 <__Z14test_array_maxv>:
100000498: d10103ff    	sub	sp, sp, #0x40
10000049c: a9037bfd    	stp	x29, x30, [sp, #0x30]
1000004a0: 9100c3fd    	add	x29, sp, #0x30
1000004a4: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
1000004a8: f9400108    	ldr	x8, [x8]
1000004ac: f9400108    	ldr	x8, [x8]
1000004b0: f81f83a8    	stur	x8, [x29, #-0x8]
1000004b4: 90000008    	adrp	x8, 0x100000000 <___stack_chk_guard+0x100000000>
1000004b8: 91166108    	add	x8, x8, #0x598
1000004bc: 3dc00100    	ldr	q0, [x8]
1000004c0: 3d8007e0    	str	q0, [sp, #0x10]
1000004c4: b9401108    	ldr	w8, [x8, #0x10]
1000004c8: b90023e8    	str	w8, [sp, #0x20]
1000004cc: b94013e8    	ldr	w8, [sp, #0x10]
1000004d0: b9000fe8    	str	w8, [sp, #0xc]
1000004d4: 52800028    	mov	w8, #0x1                ; =1
1000004d8: b9000be8    	str	w8, [sp, #0x8]
1000004dc: 14000001    	b	0x1000004e0 <__Z14test_array_maxv+0x48>
1000004e0: b9400be8    	ldr	w8, [sp, #0x8]
1000004e4: 71001508    	subs	w8, w8, #0x5
1000004e8: 5400026a    	b.ge	0x100000534 <__Z14test_array_maxv+0x9c>
1000004ec: 14000001    	b	0x1000004f0 <__Z14test_array_maxv+0x58>
1000004f0: b9800be9    	ldrsw	x9, [sp, #0x8]
1000004f4: 910043e8    	add	x8, sp, #0x10
1000004f8: b8697908    	ldr	w8, [x8, x9, lsl #2]
1000004fc: b9400fe9    	ldr	w9, [sp, #0xc]
100000500: 6b090108    	subs	w8, w8, w9
100000504: 540000ed    	b.le	0x100000520 <__Z14test_array_maxv+0x88>
100000508: 14000001    	b	0x10000050c <__Z14test_array_maxv+0x74>
10000050c: b9800be9    	ldrsw	x9, [sp, #0x8]
100000510: 910043e8    	add	x8, sp, #0x10
100000514: b8697908    	ldr	w8, [x8, x9, lsl #2]
100000518: b9000fe8    	str	w8, [sp, #0xc]
10000051c: 14000001    	b	0x100000520 <__Z14test_array_maxv+0x88>
100000520: 14000001    	b	0x100000524 <__Z14test_array_maxv+0x8c>
100000524: b9400be8    	ldr	w8, [sp, #0x8]
100000528: 11000508    	add	w8, w8, #0x1
10000052c: b9000be8    	str	w8, [sp, #0x8]
100000530: 17ffffec    	b	0x1000004e0 <__Z14test_array_maxv+0x48>
100000534: b9400fe8    	ldr	w8, [sp, #0xc]
100000538: b90007e8    	str	w8, [sp, #0x4]
10000053c: f85f83a9    	ldur	x9, [x29, #-0x8]
100000540: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
100000544: f9400108    	ldr	x8, [x8]
100000548: f9400108    	ldr	x8, [x8]
10000054c: eb090108    	subs	x8, x8, x9
100000550: 54000060    	b.eq	0x10000055c <__Z14test_array_maxv+0xc4>
100000554: 14000001    	b	0x100000558 <__Z14test_array_maxv+0xc0>
100000558: 9400000d    	bl	0x10000058c <___stack_chk_guard+0x10000058c>
10000055c: b94007e0    	ldr	w0, [sp, #0x4]
100000560: a9437bfd    	ldp	x29, x30, [sp, #0x30]
100000564: 910103ff    	add	sp, sp, #0x40
100000568: d65f03c0    	ret

000000010000056c <_main>:
10000056c: d10083ff    	sub	sp, sp, #0x20
100000570: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000574: 910043fd    	add	x29, sp, #0x10
100000578: b81fc3bf    	stur	wzr, [x29, #-0x4]
10000057c: 97ffffc7    	bl	0x100000498 <__Z14test_array_maxv>
100000580: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000584: 910083ff    	add	sp, sp, #0x20
100000588: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

000000010000058c <__stubs>:
10000058c: 90000030    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100000590: f9400610    	ldr	x16, [x16, #0x8]
100000594: d61f0200    	br	x16
