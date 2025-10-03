
/Users/jim/work/cppfort/micro-tests/results/memory/mem098-array-reverse/mem098-array-reverse_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000498 <__Z18test_array_reversev>:
100000498: d10103ff    	sub	sp, sp, #0x40
10000049c: a9037bfd    	stp	x29, x30, [sp, #0x30]
1000004a0: 9100c3fd    	add	x29, sp, #0x30
1000004a4: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
1000004a8: f9400108    	ldr	x8, [x8]
1000004ac: f9400108    	ldr	x8, [x8]
1000004b0: f81f83a8    	stur	x8, [x29, #-0x8]
1000004b4: 90000008    	adrp	x8, 0x100000000 <___stack_chk_guard+0x100000000>
1000004b8: 91165108    	add	x8, x8, #0x594
1000004bc: 3dc00100    	ldr	q0, [x8]
1000004c0: 3d8007e0    	str	q0, [sp, #0x10]
1000004c4: b9401108    	ldr	w8, [x8, #0x10]
1000004c8: b90023e8    	str	w8, [sp, #0x20]
1000004cc: b9000fff    	str	wzr, [sp, #0xc]
1000004d0: 14000001    	b	0x1000004d4 <__Z18test_array_reversev+0x3c>
1000004d4: b9400fe8    	ldr	w8, [sp, #0xc]
1000004d8: 71000908    	subs	w8, w8, #0x2
1000004dc: 540002aa    	b.ge	0x100000530 <__Z18test_array_reversev+0x98>
1000004e0: 14000001    	b	0x1000004e4 <__Z18test_array_reversev+0x4c>
1000004e4: b9800fe8    	ldrsw	x8, [sp, #0xc]
1000004e8: 910043e9    	add	x9, sp, #0x10
1000004ec: b8687928    	ldr	w8, [x9, x8, lsl #2]
1000004f0: b9000be8    	str	w8, [sp, #0x8]
1000004f4: b9400fe8    	ldr	w8, [sp, #0xc]
1000004f8: 5280008a    	mov	w10, #0x4               ; =4
1000004fc: 6b080148    	subs	w8, w10, w8
100000500: b868d928    	ldr	w8, [x9, w8, sxtw #2]
100000504: b9800feb    	ldrsw	x11, [sp, #0xc]
100000508: b82b7928    	str	w8, [x9, x11, lsl #2]
10000050c: b9400be8    	ldr	w8, [sp, #0x8]
100000510: b9400feb    	ldr	w11, [sp, #0xc]
100000514: 6b0b014a    	subs	w10, w10, w11
100000518: b82ad928    	str	w8, [x9, w10, sxtw #2]
10000051c: 14000001    	b	0x100000520 <__Z18test_array_reversev+0x88>
100000520: b9400fe8    	ldr	w8, [sp, #0xc]
100000524: 11000508    	add	w8, w8, #0x1
100000528: b9000fe8    	str	w8, [sp, #0xc]
10000052c: 17ffffea    	b	0x1000004d4 <__Z18test_array_reversev+0x3c>
100000530: b94013e8    	ldr	w8, [sp, #0x10]
100000534: b90007e8    	str	w8, [sp, #0x4]
100000538: f85f83a9    	ldur	x9, [x29, #-0x8]
10000053c: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
100000540: f9400108    	ldr	x8, [x8]
100000544: f9400108    	ldr	x8, [x8]
100000548: eb090108    	subs	x8, x8, x9
10000054c: 54000060    	b.eq	0x100000558 <__Z18test_array_reversev+0xc0>
100000550: 14000001    	b	0x100000554 <__Z18test_array_reversev+0xbc>
100000554: 9400000d    	bl	0x100000588 <___stack_chk_guard+0x100000588>
100000558: b94007e0    	ldr	w0, [sp, #0x4]
10000055c: a9437bfd    	ldp	x29, x30, [sp, #0x30]
100000560: 910103ff    	add	sp, sp, #0x40
100000564: d65f03c0    	ret

0000000100000568 <_main>:
100000568: d10083ff    	sub	sp, sp, #0x20
10000056c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000570: 910043fd    	add	x29, sp, #0x10
100000574: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000578: 97ffffc8    	bl	0x100000498 <__Z18test_array_reversev>
10000057c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000580: 910083ff    	add	sp, sp, #0x20
100000584: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

0000000100000588 <__stubs>:
100000588: 90000030    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
10000058c: f9400610    	ldr	x16, [x16, #0x8]
100000590: d61f0200    	br	x16
