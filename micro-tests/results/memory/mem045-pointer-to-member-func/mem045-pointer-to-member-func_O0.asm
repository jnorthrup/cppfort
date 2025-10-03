
/Users/jim/work/cppfort/micro-tests/results/memory/mem045-pointer-to-member-func/mem045-pointer-to-member-func_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001000003f8 <__Z27test_pointer_to_member_funcv>:
1000003f8: d10103ff    	sub	sp, sp, #0x40
1000003fc: a9037bfd    	stp	x29, x30, [sp, #0x30]
100000400: 9100c3fd    	add	x29, sp, #0x30
100000404: 90000028    	adrp	x8, 0x100004000
100000408: f9400108    	ldr	x8, [x8]
10000040c: f9000fe8    	str	x8, [sp, #0x18]
100000410: f90013ff    	str	xzr, [sp, #0x20]
100000414: f9400fe8    	ldr	x8, [sp, #0x18]
100000418: f90007e8    	str	x8, [sp, #0x8]
10000041c: f94013e8    	ldr	x8, [sp, #0x20]
100000420: 9341fd09    	asr	x9, x8, #1
100000424: f9000be9    	str	x9, [sp, #0x10]
100000428: 36000128    	tbz	w8, #0x0, 0x10000044c <__Z27test_pointer_to_member_funcv+0x54>
10000042c: 14000001    	b	0x100000430 <__Z27test_pointer_to_member_funcv+0x38>
100000430: f94007e9    	ldr	x9, [sp, #0x8]
100000434: f9400bea    	ldr	x10, [sp, #0x10]
100000438: d10007a8    	sub	x8, x29, #0x1
10000043c: f86a6908    	ldr	x8, [x8, x10]
100000440: f8694908    	ldr	x8, [x8, w9, uxtw]
100000444: f90003e8    	str	x8, [sp]
100000448: 14000004    	b	0x100000458 <__Z27test_pointer_to_member_funcv+0x60>
10000044c: f94007e8    	ldr	x8, [sp, #0x8]
100000450: f90003e8    	str	x8, [sp]
100000454: 14000001    	b	0x100000458 <__Z27test_pointer_to_member_funcv+0x60>
100000458: f9400bea    	ldr	x10, [sp, #0x10]
10000045c: f94003e8    	ldr	x8, [sp]
100000460: d10007a9    	sub	x9, x29, #0x1
100000464: 8b0a0120    	add	x0, x9, x10
100000468: 52800061    	mov	w1, #0x3                ; =3
10000046c: 52800082    	mov	w2, #0x4                ; =4
100000470: d63f0100    	blr	x8
100000474: a9437bfd    	ldp	x29, x30, [sp, #0x30]
100000478: 910103ff    	add	sp, sp, #0x40
10000047c: d65f03c0    	ret

0000000100000480 <__ZN10Calculator3addEii>:
100000480: d10043ff    	sub	sp, sp, #0x10
100000484: f90007e0    	str	x0, [sp, #0x8]
100000488: b90007e1    	str	w1, [sp, #0x4]
10000048c: b90003e2    	str	w2, [sp]
100000490: b94007e8    	ldr	w8, [sp, #0x4]
100000494: b94003e9    	ldr	w9, [sp]
100000498: 0b090100    	add	w0, w8, w9
10000049c: 910043ff    	add	sp, sp, #0x10
1000004a0: d65f03c0    	ret

00000001000004a4 <_main>:
1000004a4: d10083ff    	sub	sp, sp, #0x20
1000004a8: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000004ac: 910043fd    	add	x29, sp, #0x10
1000004b0: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000004b4: 97ffffd1    	bl	0x1000003f8 <__Z27test_pointer_to_member_funcv>
1000004b8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000004bc: 910083ff    	add	sp, sp, #0x20
1000004c0: d65f03c0    	ret
