
/Users/jim/work/cppfort/micro-tests/results/memory/mem026-array-of-pointers/mem026-array-of-pointers_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000448 <__Z22test_array_of_pointersv>:
100000448: d10103ff    	sub	sp, sp, #0x40
10000044c: a9037bfd    	stp	x29, x30, [sp, #0x30]
100000450: 9100c3fd    	add	x29, sp, #0x30
100000454: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
100000458: f9400108    	ldr	x8, [x8]
10000045c: f9400108    	ldr	x8, [x8]
100000460: f81f83a8    	stur	x8, [x29, #-0x8]
100000464: 910033ea    	add	x10, sp, #0xc
100000468: 52800028    	mov	w8, #0x1                ; =1
10000046c: b9000fe8    	str	w8, [sp, #0xc]
100000470: 910023e9    	add	x9, sp, #0x8
100000474: 52800048    	mov	w8, #0x2                ; =2
100000478: b9000be8    	str	w8, [sp, #0x8]
10000047c: 910013e8    	add	x8, sp, #0x4
100000480: 5280006b    	mov	w11, #0x3               ; =3
100000484: b90007eb    	str	w11, [sp, #0x4]
100000488: f9000bea    	str	x10, [sp, #0x10]
10000048c: f9000fe9    	str	x9, [sp, #0x18]
100000490: f90013e8    	str	x8, [sp, #0x20]
100000494: f9400fe8    	ldr	x8, [sp, #0x18]
100000498: b9400108    	ldr	w8, [x8]
10000049c: b90003e8    	str	w8, [sp]
1000004a0: f85f83a9    	ldur	x9, [x29, #-0x8]
1000004a4: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
1000004a8: f9400108    	ldr	x8, [x8]
1000004ac: f9400108    	ldr	x8, [x8]
1000004b0: eb090108    	subs	x8, x8, x9
1000004b4: 54000060    	b.eq	0x1000004c0 <__Z22test_array_of_pointersv+0x78>
1000004b8: 14000001    	b	0x1000004bc <__Z22test_array_of_pointersv+0x74>
1000004bc: 9400000d    	bl	0x1000004f0 <___stack_chk_guard+0x1000004f0>
1000004c0: b94003e0    	ldr	w0, [sp]
1000004c4: a9437bfd    	ldp	x29, x30, [sp, #0x30]
1000004c8: 910103ff    	add	sp, sp, #0x40
1000004cc: d65f03c0    	ret

00000001000004d0 <_main>:
1000004d0: d10083ff    	sub	sp, sp, #0x20
1000004d4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000004d8: 910043fd    	add	x29, sp, #0x10
1000004dc: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000004e0: 97ffffda    	bl	0x100000448 <__Z22test_array_of_pointersv>
1000004e4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000004e8: 910083ff    	add	sp, sp, #0x20
1000004ec: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

00000001000004f0 <__stubs>:
1000004f0: 90000030    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000004f4: f9400610    	ldr	x16, [x16, #0x8]
1000004f8: d61f0200    	br	x16
