
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf027-range-for-const-ref/cf027-range-for-const-ref_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000498 <__Z24test_range_for_const_refv>:
100000498: d10183ff    	sub	sp, sp, #0x60
10000049c: a9057bfd    	stp	x29, x30, [sp, #0x50]
1000004a0: 910143fd    	add	x29, sp, #0x50
1000004a4: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
1000004a8: f9400108    	ldr	x8, [x8]
1000004ac: f9400108    	ldr	x8, [x8]
1000004b0: f81f83a8    	stur	x8, [x29, #-0x8]
1000004b4: 90000009    	adrp	x9, 0x100000000 <___stack_chk_guard+0x100000000>
1000004b8: 91166129    	add	x9, x9, #0x598
1000004bc: 3dc00120    	ldr	q0, [x9]
1000004c0: d10083a8    	sub	x8, x29, #0x20
1000004c4: 3c9e03a0    	stur	q0, [x29, #-0x20]
1000004c8: b9401129    	ldr	w9, [x9, #0x10]
1000004cc: b81f03a9    	stur	w9, [x29, #-0x10]
1000004d0: b81dc3bf    	stur	wzr, [x29, #-0x24]
1000004d4: f90013e8    	str	x8, [sp, #0x20]
1000004d8: f94013e8    	ldr	x8, [sp, #0x20]
1000004dc: f9000fe8    	str	x8, [sp, #0x18]
1000004e0: f94013e8    	ldr	x8, [sp, #0x20]
1000004e4: 91005108    	add	x8, x8, #0x14
1000004e8: f9000be8    	str	x8, [sp, #0x10]
1000004ec: 14000001    	b	0x1000004f0 <__Z24test_range_for_const_refv+0x58>
1000004f0: f9400fe8    	ldr	x8, [sp, #0x18]
1000004f4: f9400be9    	ldr	x9, [sp, #0x10]
1000004f8: eb090108    	subs	x8, x8, x9
1000004fc: 540001c0    	b.eq	0x100000534 <__Z24test_range_for_const_refv+0x9c>
100000500: 14000001    	b	0x100000504 <__Z24test_range_for_const_refv+0x6c>
100000504: f9400fe8    	ldr	x8, [sp, #0x18]
100000508: f90007e8    	str	x8, [sp, #0x8]
10000050c: f94007e8    	ldr	x8, [sp, #0x8]
100000510: b9400109    	ldr	w9, [x8]
100000514: b85dc3a8    	ldur	w8, [x29, #-0x24]
100000518: 0b090108    	add	w8, w8, w9
10000051c: b81dc3a8    	stur	w8, [x29, #-0x24]
100000520: 14000001    	b	0x100000524 <__Z24test_range_for_const_refv+0x8c>
100000524: f9400fe8    	ldr	x8, [sp, #0x18]
100000528: 91001108    	add	x8, x8, #0x4
10000052c: f9000fe8    	str	x8, [sp, #0x18]
100000530: 17fffff0    	b	0x1000004f0 <__Z24test_range_for_const_refv+0x58>
100000534: b85dc3a8    	ldur	w8, [x29, #-0x24]
100000538: b90007e8    	str	w8, [sp, #0x4]
10000053c: f85f83a9    	ldur	x9, [x29, #-0x8]
100000540: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
100000544: f9400108    	ldr	x8, [x8]
100000548: f9400108    	ldr	x8, [x8]
10000054c: eb090108    	subs	x8, x8, x9
100000550: 54000060    	b.eq	0x10000055c <__Z24test_range_for_const_refv+0xc4>
100000554: 14000001    	b	0x100000558 <__Z24test_range_for_const_refv+0xc0>
100000558: 9400000d    	bl	0x10000058c <___stack_chk_guard+0x10000058c>
10000055c: b94007e0    	ldr	w0, [sp, #0x4]
100000560: a9457bfd    	ldp	x29, x30, [sp, #0x50]
100000564: 910183ff    	add	sp, sp, #0x60
100000568: d65f03c0    	ret

000000010000056c <_main>:
10000056c: d10083ff    	sub	sp, sp, #0x20
100000570: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000574: 910043fd    	add	x29, sp, #0x10
100000578: b81fc3bf    	stur	wzr, [x29, #-0x4]
10000057c: 97ffffc7    	bl	0x100000498 <__Z24test_range_for_const_refv>
100000580: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000584: 910083ff    	add	sp, sp, #0x20
100000588: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

000000010000058c <__stubs>:
10000058c: 90000030    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100000590: f9400610    	ldr	x16, [x16, #0x8]
100000594: d61f0200    	br	x16
