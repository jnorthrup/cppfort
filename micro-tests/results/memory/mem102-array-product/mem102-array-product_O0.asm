
/Users/jim/work/cppfort/micro-tests/results/memory/mem102-array-product/mem102-array-product_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000498 <__Z18test_array_productv>:
100000498: d10103ff    	sub	sp, sp, #0x40
10000049c: a9037bfd    	stp	x29, x30, [sp, #0x30]
1000004a0: 9100c3fd    	add	x29, sp, #0x30
1000004a4: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
1000004a8: f9400108    	ldr	x8, [x8]
1000004ac: f9400108    	ldr	x8, [x8]
1000004b0: f81f83a8    	stur	x8, [x29, #-0x8]
1000004b4: 90000008    	adrp	x8, 0x100000000 <___stack_chk_guard+0x100000000>
1000004b8: 91160108    	add	x8, x8, #0x580
1000004bc: 3dc00100    	ldr	q0, [x8]
1000004c0: 3d8007e0    	str	q0, [sp, #0x10]
1000004c4: 52800028    	mov	w8, #0x1                ; =1
1000004c8: b9000fe8    	str	w8, [sp, #0xc]
1000004cc: b9000bff    	str	wzr, [sp, #0x8]
1000004d0: 14000001    	b	0x1000004d4 <__Z18test_array_productv+0x3c>
1000004d4: b9400be8    	ldr	w8, [sp, #0x8]
1000004d8: 71001108    	subs	w8, w8, #0x4
1000004dc: 540001aa    	b.ge	0x100000510 <__Z18test_array_productv+0x78>
1000004e0: 14000001    	b	0x1000004e4 <__Z18test_array_productv+0x4c>
1000004e4: b9800be9    	ldrsw	x9, [sp, #0x8]
1000004e8: 910043e8    	add	x8, sp, #0x10
1000004ec: b8697909    	ldr	w9, [x8, x9, lsl #2]
1000004f0: b9400fe8    	ldr	w8, [sp, #0xc]
1000004f4: 1b097d08    	mul	w8, w8, w9
1000004f8: b9000fe8    	str	w8, [sp, #0xc]
1000004fc: 14000001    	b	0x100000500 <__Z18test_array_productv+0x68>
100000500: b9400be8    	ldr	w8, [sp, #0x8]
100000504: 11000508    	add	w8, w8, #0x1
100000508: b9000be8    	str	w8, [sp, #0x8]
10000050c: 17fffff2    	b	0x1000004d4 <__Z18test_array_productv+0x3c>
100000510: b9400fe8    	ldr	w8, [sp, #0xc]
100000514: b90007e8    	str	w8, [sp, #0x4]
100000518: f85f83a9    	ldur	x9, [x29, #-0x8]
10000051c: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
100000520: f9400108    	ldr	x8, [x8]
100000524: f9400108    	ldr	x8, [x8]
100000528: eb090108    	subs	x8, x8, x9
10000052c: 54000060    	b.eq	0x100000538 <__Z18test_array_productv+0xa0>
100000530: 14000001    	b	0x100000534 <__Z18test_array_productv+0x9c>
100000534: 9400000d    	bl	0x100000568 <___stack_chk_guard+0x100000568>
100000538: b94007e0    	ldr	w0, [sp, #0x4]
10000053c: a9437bfd    	ldp	x29, x30, [sp, #0x30]
100000540: 910103ff    	add	sp, sp, #0x40
100000544: d65f03c0    	ret

0000000100000548 <_main>:
100000548: d10083ff    	sub	sp, sp, #0x20
10000054c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000550: 910043fd    	add	x29, sp, #0x10
100000554: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000558: 97ffffd0    	bl	0x100000498 <__Z18test_array_productv>
10000055c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000560: 910083ff    	add	sp, sp, #0x20
100000564: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

0000000100000568 <__stubs>:
100000568: 90000030    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
10000056c: f9400610    	ldr	x16, [x16, #0x8]
100000570: d61f0200    	br	x16
