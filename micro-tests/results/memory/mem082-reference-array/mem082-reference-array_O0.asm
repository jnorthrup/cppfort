
/Users/jim/work/cppfort/micro-tests/results/memory/mem082-reference-array/mem082-reference-array_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000498 <__Z20test_reference_arrayv>:
100000498: d10103ff    	sub	sp, sp, #0x40
10000049c: a9037bfd    	stp	x29, x30, [sp, #0x30]
1000004a0: 9100c3fd    	add	x29, sp, #0x30
1000004a4: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
1000004a8: f9400108    	ldr	x8, [x8]
1000004ac: f9400108    	ldr	x8, [x8]
1000004b0: f81f83a8    	stur	x8, [x29, #-0x8]
1000004b4: 90000009    	adrp	x9, 0x100000000 <___stack_chk_guard+0x100000000>
1000004b8: 9114f129    	add	x9, x9, #0x53c
1000004bc: f940012a    	ldr	x10, [x9]
1000004c0: 910063e8    	add	x8, sp, #0x18
1000004c4: f9000fea    	str	x10, [sp, #0x18]
1000004c8: b9400929    	ldr	w9, [x9, #0x8]
1000004cc: b90023e9    	str	w9, [sp, #0x20]
1000004d0: f9000be8    	str	x8, [sp, #0x10]
1000004d4: f9400be8    	ldr	x8, [sp, #0x10]
1000004d8: b9400508    	ldr	w8, [x8, #0x4]
1000004dc: b9000fe8    	str	w8, [sp, #0xc]
1000004e0: f85f83a9    	ldur	x9, [x29, #-0x8]
1000004e4: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
1000004e8: f9400108    	ldr	x8, [x8]
1000004ec: f9400108    	ldr	x8, [x8]
1000004f0: eb090108    	subs	x8, x8, x9
1000004f4: 54000060    	b.eq	0x100000500 <__Z20test_reference_arrayv+0x68>
1000004f8: 14000001    	b	0x1000004fc <__Z20test_reference_arrayv+0x64>
1000004fc: 9400000d    	bl	0x100000530 <___stack_chk_guard+0x100000530>
100000500: b9400fe0    	ldr	w0, [sp, #0xc]
100000504: a9437bfd    	ldp	x29, x30, [sp, #0x30]
100000508: 910103ff    	add	sp, sp, #0x40
10000050c: d65f03c0    	ret

0000000100000510 <_main>:
100000510: d10083ff    	sub	sp, sp, #0x20
100000514: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000518: 910043fd    	add	x29, sp, #0x10
10000051c: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000520: 97ffffde    	bl	0x100000498 <__Z20test_reference_arrayv>
100000524: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000528: 910083ff    	add	sp, sp, #0x20
10000052c: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

0000000100000530 <__stubs>:
100000530: 90000030    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100000534: f9400610    	ldr	x16, [x16, #0x8]
100000538: d61f0200    	br	x16
