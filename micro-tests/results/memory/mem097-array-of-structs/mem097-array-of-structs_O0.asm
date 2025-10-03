
/Users/jim/work/cppfort/micro-tests/results/memory/mem097-array-of-structs/mem097-array-of-structs_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000498 <__Z21test_array_of_structsv>:
100000498: d10103ff    	sub	sp, sp, #0x40
10000049c: a9037bfd    	stp	x29, x30, [sp, #0x30]
1000004a0: 9100c3fd    	add	x29, sp, #0x30
1000004a4: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
1000004a8: f9400108    	ldr	x8, [x8]
1000004ac: f9400108    	ldr	x8, [x8]
1000004b0: f81f83a8    	stur	x8, [x29, #-0x8]
1000004b4: 90000008    	adrp	x8, 0x100000000 <___stack_chk_guard+0x100000000>
1000004b8: 9114e108    	add	x8, x8, #0x538
1000004bc: 3dc00100    	ldr	q0, [x8]
1000004c0: 3d8007e0    	str	q0, [sp, #0x10]
1000004c4: f9400908    	ldr	x8, [x8, #0x10]
1000004c8: f90013e8    	str	x8, [sp, #0x20]
1000004cc: b9401be8    	ldr	w8, [sp, #0x18]
1000004d0: b9401fe9    	ldr	w9, [sp, #0x1c]
1000004d4: 0b090108    	add	w8, w8, w9
1000004d8: b9000fe8    	str	w8, [sp, #0xc]
1000004dc: f85f83a9    	ldur	x9, [x29, #-0x8]
1000004e0: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
1000004e4: f9400108    	ldr	x8, [x8]
1000004e8: f9400108    	ldr	x8, [x8]
1000004ec: eb090108    	subs	x8, x8, x9
1000004f0: 54000060    	b.eq	0x1000004fc <__Z21test_array_of_structsv+0x64>
1000004f4: 14000001    	b	0x1000004f8 <__Z21test_array_of_structsv+0x60>
1000004f8: 9400000d    	bl	0x10000052c <___stack_chk_guard+0x10000052c>
1000004fc: b9400fe0    	ldr	w0, [sp, #0xc]
100000500: a9437bfd    	ldp	x29, x30, [sp, #0x30]
100000504: 910103ff    	add	sp, sp, #0x40
100000508: d65f03c0    	ret

000000010000050c <_main>:
10000050c: d10083ff    	sub	sp, sp, #0x20
100000510: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000514: 910043fd    	add	x29, sp, #0x10
100000518: b81fc3bf    	stur	wzr, [x29, #-0x4]
10000051c: 97ffffdf    	bl	0x100000498 <__Z21test_array_of_structsv>
100000520: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000524: 910083ff    	add	sp, sp, #0x20
100000528: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

000000010000052c <__stubs>:
10000052c: 90000030    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100000530: f9400610    	ldr	x16, [x16, #0x8]
100000534: d61f0200    	br	x16
