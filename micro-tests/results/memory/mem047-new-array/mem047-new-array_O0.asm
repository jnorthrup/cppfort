
/Users/jim/work/cppfort/micro-tests/results/memory/mem047-new-array/mem047-new-array_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000448 <__Z14test_new_arrayv>:
100000448: d100c3ff    	sub	sp, sp, #0x30
10000044c: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000450: 910083fd    	add	x29, sp, #0x20
100000454: d2800280    	mov	x0, #0x14               ; =20
100000458: 94000022    	bl	0x1000004e0 <__Znam+0x1000004e0>
10000045c: 52800028    	mov	w8, #0x1                ; =1
100000460: b9000008    	str	w8, [x0]
100000464: 52800048    	mov	w8, #0x2                ; =2
100000468: b9000408    	str	w8, [x0, #0x4]
10000046c: 52800068    	mov	w8, #0x3                ; =3
100000470: b9000808    	str	w8, [x0, #0x8]
100000474: 52800088    	mov	w8, #0x4                ; =4
100000478: b9000c08    	str	w8, [x0, #0xc]
10000047c: 528000a8    	mov	w8, #0x5                ; =5
100000480: b9001008    	str	w8, [x0, #0x10]
100000484: f81f83a0    	stur	x0, [x29, #-0x8]
100000488: f85f83a8    	ldur	x8, [x29, #-0x8]
10000048c: b9400908    	ldr	w8, [x8, #0x8]
100000490: b81f43a8    	stur	w8, [x29, #-0xc]
100000494: f85f83a8    	ldur	x8, [x29, #-0x8]
100000498: f90007e8    	str	x8, [sp, #0x8]
10000049c: b40000a8    	cbz	x8, 0x1000004b0 <__Z14test_new_arrayv+0x68>
1000004a0: 14000001    	b	0x1000004a4 <__Z14test_new_arrayv+0x5c>
1000004a4: f94007e0    	ldr	x0, [sp, #0x8]
1000004a8: 94000011    	bl	0x1000004ec <__Znam+0x1000004ec>
1000004ac: 14000001    	b	0x1000004b0 <__Z14test_new_arrayv+0x68>
1000004b0: b85f43a0    	ldur	w0, [x29, #-0xc]
1000004b4: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000004b8: 9100c3ff    	add	sp, sp, #0x30
1000004bc: d65f03c0    	ret

00000001000004c0 <_main>:
1000004c0: d10083ff    	sub	sp, sp, #0x20
1000004c4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000004c8: 910043fd    	add	x29, sp, #0x10
1000004cc: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000004d0: 97ffffde    	bl	0x100000448 <__Z14test_new_arrayv>
1000004d4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000004d8: 910083ff    	add	sp, sp, #0x20
1000004dc: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

00000001000004e0 <__stubs>:
1000004e0: 90000030    	adrp	x16, 0x100004000 <__Znam+0x100004000>
1000004e4: f9400210    	ldr	x16, [x16]
1000004e8: d61f0200    	br	x16
1000004ec: 90000030    	adrp	x16, 0x100004000 <__Znam+0x100004000>
1000004f0: f9400610    	ldr	x16, [x16, #0x8]
1000004f4: d61f0200    	br	x16
