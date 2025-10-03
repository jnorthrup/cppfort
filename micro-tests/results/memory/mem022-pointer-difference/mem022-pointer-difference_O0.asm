
/Users/jim/work/cppfort/micro-tests/results/memory/mem022-pointer-difference/mem022-pointer-difference_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000498 <__Z23test_pointer_differencev>:
100000498: d10143ff    	sub	sp, sp, #0x50
10000049c: a9047bfd    	stp	x29, x30, [sp, #0x40]
1000004a0: 910103fd    	add	x29, sp, #0x40
1000004a4: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
1000004a8: f9400108    	ldr	x8, [x8]
1000004ac: f9400108    	ldr	x8, [x8]
1000004b0: f81f83a8    	stur	x8, [x29, #-0x8]
1000004b4: 90000009    	adrp	x9, 0x100000000 <___stack_chk_guard+0x100000000>
1000004b8: 91156129    	add	x9, x9, #0x558
1000004bc: 3dc00120    	ldr	q0, [x9]
1000004c0: 910083e8    	add	x8, sp, #0x20
1000004c4: 3d800be0    	str	q0, [sp, #0x20]
1000004c8: b9401129    	ldr	w9, [x9, #0x10]
1000004cc: b90033e9    	str	w9, [sp, #0x30]
1000004d0: aa0803e9    	mov	x9, x8
1000004d4: f9000fe9    	str	x9, [sp, #0x18]
1000004d8: 91003108    	add	x8, x8, #0xc
1000004dc: f9000be8    	str	x8, [sp, #0x10]
1000004e0: f9400be8    	ldr	x8, [sp, #0x10]
1000004e4: f9400fe9    	ldr	x9, [sp, #0x18]
1000004e8: eb090108    	subs	x8, x8, x9
1000004ec: d2800089    	mov	x9, #0x4                ; =4
1000004f0: 9ac90d08    	sdiv	x8, x8, x9
1000004f4: f90007e8    	str	x8, [sp, #0x8]
1000004f8: f85f83a9    	ldur	x9, [x29, #-0x8]
1000004fc: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
100000500: f9400108    	ldr	x8, [x8]
100000504: f9400108    	ldr	x8, [x8]
100000508: eb090108    	subs	x8, x8, x9
10000050c: 54000060    	b.eq	0x100000518 <__Z23test_pointer_differencev+0x80>
100000510: 14000001    	b	0x100000514 <__Z23test_pointer_differencev+0x7c>
100000514: 9400000e    	bl	0x10000054c <___stack_chk_guard+0x10000054c>
100000518: f94007e8    	ldr	x8, [sp, #0x8]
10000051c: aa0803e0    	mov	x0, x8
100000520: a9447bfd    	ldp	x29, x30, [sp, #0x40]
100000524: 910143ff    	add	sp, sp, #0x50
100000528: d65f03c0    	ret

000000010000052c <_main>:
10000052c: d10083ff    	sub	sp, sp, #0x20
100000530: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000534: 910043fd    	add	x29, sp, #0x10
100000538: b81fc3bf    	stur	wzr, [x29, #-0x4]
10000053c: 97ffffd7    	bl	0x100000498 <__Z23test_pointer_differencev>
100000540: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000544: 910083ff    	add	sp, sp, #0x20
100000548: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

000000010000054c <__stubs>:
10000054c: 90000030    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100000550: f9400610    	ldr	x16, [x16, #0x8]
100000554: d61f0200    	br	x16
