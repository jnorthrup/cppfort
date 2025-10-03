
/Users/jim/work/cppfort/micro-tests/results/memory/mem091-array-bounds/mem091-array-bounds_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000498 <__Z17test_array_boundsv>:
100000498: d10103ff    	sub	sp, sp, #0x40
10000049c: a9037bfd    	stp	x29, x30, [sp, #0x30]
1000004a0: 9100c3fd    	add	x29, sp, #0x30
1000004a4: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
1000004a8: f9400108    	ldr	x8, [x8]
1000004ac: f9400108    	ldr	x8, [x8]
1000004b0: f81f83a8    	stur	x8, [x29, #-0x8]
1000004b4: 90000008    	adrp	x8, 0x100000000 <___stack_chk_guard+0x100000000>
1000004b8: 9114c108    	add	x8, x8, #0x530
1000004bc: 3dc00100    	ldr	q0, [x8]
1000004c0: 3d8007e0    	str	q0, [sp, #0x10]
1000004c4: b9401108    	ldr	w8, [x8, #0x10]
1000004c8: b90023e8    	str	w8, [sp, #0x20]
1000004cc: b94023e8    	ldr	w8, [sp, #0x20]
1000004d0: b9000fe8    	str	w8, [sp, #0xc]
1000004d4: f85f83a9    	ldur	x9, [x29, #-0x8]
1000004d8: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
1000004dc: f9400108    	ldr	x8, [x8]
1000004e0: f9400108    	ldr	x8, [x8]
1000004e4: eb090108    	subs	x8, x8, x9
1000004e8: 54000060    	b.eq	0x1000004f4 <__Z17test_array_boundsv+0x5c>
1000004ec: 14000001    	b	0x1000004f0 <__Z17test_array_boundsv+0x58>
1000004f0: 9400000d    	bl	0x100000524 <___stack_chk_guard+0x100000524>
1000004f4: b9400fe0    	ldr	w0, [sp, #0xc]
1000004f8: a9437bfd    	ldp	x29, x30, [sp, #0x30]
1000004fc: 910103ff    	add	sp, sp, #0x40
100000500: d65f03c0    	ret

0000000100000504 <_main>:
100000504: d10083ff    	sub	sp, sp, #0x20
100000508: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000050c: 910043fd    	add	x29, sp, #0x10
100000510: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000514: 97ffffe1    	bl	0x100000498 <__Z17test_array_boundsv>
100000518: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000051c: 910083ff    	add	sp, sp, #0x20
100000520: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

0000000100000524 <__stubs>:
100000524: 90000030    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100000528: f9400610    	ldr	x16, [x16, #0x8]
10000052c: d61f0200    	br	x16
