
/Users/jim/work/cppfort/micro-tests/results/memory/mem087-array-zero-init/mem087-array-zero-init_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000448 <__Z20test_array_zero_initv>:
100000448: d10103ff    	sub	sp, sp, #0x40
10000044c: a9037bfd    	stp	x29, x30, [sp, #0x30]
100000450: 9100c3fd    	add	x29, sp, #0x30
100000454: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
100000458: f9400108    	ldr	x8, [x8]
10000045c: f9400108    	ldr	x8, [x8]
100000460: f81f83a8    	stur	x8, [x29, #-0x8]
100000464: f9000bff    	str	xzr, [sp, #0x10]
100000468: f9000fff    	str	xzr, [sp, #0x18]
10000046c: b90023ff    	str	wzr, [sp, #0x20]
100000470: b94013e8    	ldr	w8, [sp, #0x10]
100000474: b94023e9    	ldr	w9, [sp, #0x20]
100000478: 0b090108    	add	w8, w8, w9
10000047c: b9000fe8    	str	w8, [sp, #0xc]
100000480: f85f83a9    	ldur	x9, [x29, #-0x8]
100000484: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
100000488: f9400108    	ldr	x8, [x8]
10000048c: f9400108    	ldr	x8, [x8]
100000490: eb090108    	subs	x8, x8, x9
100000494: 54000060    	b.eq	0x1000004a0 <__Z20test_array_zero_initv+0x58>
100000498: 14000001    	b	0x10000049c <__Z20test_array_zero_initv+0x54>
10000049c: 9400000d    	bl	0x1000004d0 <___stack_chk_guard+0x1000004d0>
1000004a0: b9400fe0    	ldr	w0, [sp, #0xc]
1000004a4: a9437bfd    	ldp	x29, x30, [sp, #0x30]
1000004a8: 910103ff    	add	sp, sp, #0x40
1000004ac: d65f03c0    	ret

00000001000004b0 <_main>:
1000004b0: d10083ff    	sub	sp, sp, #0x20
1000004b4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000004b8: 910043fd    	add	x29, sp, #0x10
1000004bc: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000004c0: 97ffffe2    	bl	0x100000448 <__Z20test_array_zero_initv>
1000004c4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000004c8: 910083ff    	add	sp, sp, #0x20
1000004cc: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

00000001000004d0 <__stubs>:
1000004d0: 90000030    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000004d4: f9400610    	ldr	x16, [x16, #0x8]
1000004d8: d61f0200    	br	x16
