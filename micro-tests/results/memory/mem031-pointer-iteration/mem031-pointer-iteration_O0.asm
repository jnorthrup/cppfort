
/Users/jim/work/cppfort/micro-tests/results/memory/mem031-pointer-iteration/mem031-pointer-iteration_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000498 <__Z22test_pointer_iterationv>:
100000498: d10143ff    	sub	sp, sp, #0x50
10000049c: a9047bfd    	stp	x29, x30, [sp, #0x40]
1000004a0: 910103fd    	add	x29, sp, #0x40
1000004a4: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
1000004a8: f9400108    	ldr	x8, [x8]
1000004ac: f9400108    	ldr	x8, [x8]
1000004b0: f81f83a8    	stur	x8, [x29, #-0x8]
1000004b4: 90000009    	adrp	x9, 0x100000000 <___stack_chk_guard+0x100000000>
1000004b8: 91160129    	add	x9, x9, #0x580
1000004bc: 3dc00120    	ldr	q0, [x9]
1000004c0: 910083e8    	add	x8, sp, #0x20
1000004c4: 3d800be0    	str	q0, [sp, #0x20]
1000004c8: b9401129    	ldr	w9, [x9, #0x10]
1000004cc: b90033e9    	str	w9, [sp, #0x30]
1000004d0: b9001fff    	str	wzr, [sp, #0x1c]
1000004d4: f9000be8    	str	x8, [sp, #0x10]
1000004d8: 14000001    	b	0x1000004dc <__Z22test_pointer_iterationv+0x44>
1000004dc: f9400be8    	ldr	x8, [sp, #0x10]
1000004e0: 910083e9    	add	x9, sp, #0x20
1000004e4: 91005129    	add	x9, x9, #0x14
1000004e8: eb090108    	subs	x8, x8, x9
1000004ec: 54000182    	b.hs	0x10000051c <__Z22test_pointer_iterationv+0x84>
1000004f0: 14000001    	b	0x1000004f4 <__Z22test_pointer_iterationv+0x5c>
1000004f4: f9400be8    	ldr	x8, [sp, #0x10]
1000004f8: b9400109    	ldr	w9, [x8]
1000004fc: b9401fe8    	ldr	w8, [sp, #0x1c]
100000500: 0b090108    	add	w8, w8, w9
100000504: b9001fe8    	str	w8, [sp, #0x1c]
100000508: 14000001    	b	0x10000050c <__Z22test_pointer_iterationv+0x74>
10000050c: f9400be8    	ldr	x8, [sp, #0x10]
100000510: 91001108    	add	x8, x8, #0x4
100000514: f9000be8    	str	x8, [sp, #0x10]
100000518: 17fffff1    	b	0x1000004dc <__Z22test_pointer_iterationv+0x44>
10000051c: b9401fe8    	ldr	w8, [sp, #0x1c]
100000520: b9000fe8    	str	w8, [sp, #0xc]
100000524: f85f83a9    	ldur	x9, [x29, #-0x8]
100000528: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
10000052c: f9400108    	ldr	x8, [x8]
100000530: f9400108    	ldr	x8, [x8]
100000534: eb090108    	subs	x8, x8, x9
100000538: 54000060    	b.eq	0x100000544 <__Z22test_pointer_iterationv+0xac>
10000053c: 14000001    	b	0x100000540 <__Z22test_pointer_iterationv+0xa8>
100000540: 9400000d    	bl	0x100000574 <___stack_chk_guard+0x100000574>
100000544: b9400fe0    	ldr	w0, [sp, #0xc]
100000548: a9447bfd    	ldp	x29, x30, [sp, #0x40]
10000054c: 910143ff    	add	sp, sp, #0x50
100000550: d65f03c0    	ret

0000000100000554 <_main>:
100000554: d10083ff    	sub	sp, sp, #0x20
100000558: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000055c: 910043fd    	add	x29, sp, #0x10
100000560: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000564: 97ffffcd    	bl	0x100000498 <__Z22test_pointer_iterationv>
100000568: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000056c: 910083ff    	add	sp, sp, #0x20
100000570: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

0000000100000574 <__stubs>:
100000574: 90000030    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100000578: f9400610    	ldr	x16, [x16, #0x8]
10000057c: d61f0200    	br	x16
