
/Users/jim/work/cppfort/micro-tests/results/memory/mem023-pointer-comparison/mem023-pointer-comparison_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000498 <__Z23test_pointer_comparisonv>:
100000498: d10143ff    	sub	sp, sp, #0x50
10000049c: a9047bfd    	stp	x29, x30, [sp, #0x40]
1000004a0: 910103fd    	add	x29, sp, #0x40
1000004a4: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
1000004a8: f9400108    	ldr	x8, [x8]
1000004ac: f9400108    	ldr	x8, [x8]
1000004b0: f81f83a8    	stur	x8, [x29, #-0x8]
1000004b4: 90000009    	adrp	x9, 0x100000000 <___stack_chk_guard+0x100000000>
1000004b8: 91154129    	add	x9, x9, #0x550
1000004bc: 3dc00120    	ldr	q0, [x9]
1000004c0: 910083e8    	add	x8, sp, #0x20
1000004c4: 3d800be0    	str	q0, [sp, #0x20]
1000004c8: b9401129    	ldr	w9, [x9, #0x10]
1000004cc: b90033e9    	str	w9, [sp, #0x30]
1000004d0: aa0803e9    	mov	x9, x8
1000004d4: f9000fe9    	str	x9, [sp, #0x18]
1000004d8: 91003108    	add	x8, x8, #0xc
1000004dc: f9000be8    	str	x8, [sp, #0x10]
1000004e0: f9400fe8    	ldr	x8, [sp, #0x18]
1000004e4: f9400be9    	ldr	x9, [sp, #0x10]
1000004e8: eb090108    	subs	x8, x8, x9
1000004ec: 1a9f27e8    	cset	w8, lo
1000004f0: b9000fe8    	str	w8, [sp, #0xc]
1000004f4: f85f83a9    	ldur	x9, [x29, #-0x8]
1000004f8: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
1000004fc: f9400108    	ldr	x8, [x8]
100000500: f9400108    	ldr	x8, [x8]
100000504: eb090108    	subs	x8, x8, x9
100000508: 54000060    	b.eq	0x100000514 <__Z23test_pointer_comparisonv+0x7c>
10000050c: 14000001    	b	0x100000510 <__Z23test_pointer_comparisonv+0x78>
100000510: 9400000d    	bl	0x100000544 <___stack_chk_guard+0x100000544>
100000514: b9400fe0    	ldr	w0, [sp, #0xc]
100000518: a9447bfd    	ldp	x29, x30, [sp, #0x40]
10000051c: 910143ff    	add	sp, sp, #0x50
100000520: d65f03c0    	ret

0000000100000524 <_main>:
100000524: d10083ff    	sub	sp, sp, #0x20
100000528: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000052c: 910043fd    	add	x29, sp, #0x10
100000530: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000534: 97ffffd9    	bl	0x100000498 <__Z23test_pointer_comparisonv>
100000538: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000053c: 910083ff    	add	sp, sp, #0x20
100000540: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

0000000100000544 <__stubs>:
100000544: 90000030    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100000548: f9400610    	ldr	x16, [x16, #0x8]
10000054c: d61f0200    	br	x16
