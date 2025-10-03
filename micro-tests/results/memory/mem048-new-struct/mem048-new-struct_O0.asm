
/Users/jim/work/cppfort/micro-tests/results/memory/mem048-new-struct/mem048-new-struct_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000448 <__Z15test_new_structv>:
100000448: d100c3ff    	sub	sp, sp, #0x30
10000044c: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000450: 910083fd    	add	x29, sp, #0x20
100000454: d2800100    	mov	x0, #0x8                ; =8
100000458: 9400001f    	bl	0x1000004d4 <__Znwm+0x1000004d4>
10000045c: 52800068    	mov	w8, #0x3                ; =3
100000460: b9000008    	str	w8, [x0]
100000464: 52800088    	mov	w8, #0x4                ; =4
100000468: b9000408    	str	w8, [x0, #0x4]
10000046c: f81f83a0    	stur	x0, [x29, #-0x8]
100000470: f85f83a8    	ldur	x8, [x29, #-0x8]
100000474: b9400108    	ldr	w8, [x8]
100000478: f85f83a9    	ldur	x9, [x29, #-0x8]
10000047c: b9400529    	ldr	w9, [x9, #0x4]
100000480: 0b090108    	add	w8, w8, w9
100000484: b81f43a8    	stur	w8, [x29, #-0xc]
100000488: f85f83a8    	ldur	x8, [x29, #-0x8]
10000048c: f90007e8    	str	x8, [sp, #0x8]
100000490: b40000a8    	cbz	x8, 0x1000004a4 <__Z15test_new_structv+0x5c>
100000494: 14000001    	b	0x100000498 <__Z15test_new_structv+0x50>
100000498: f94007e0    	ldr	x0, [sp, #0x8]
10000049c: 94000011    	bl	0x1000004e0 <__Znwm+0x1000004e0>
1000004a0: 14000001    	b	0x1000004a4 <__Z15test_new_structv+0x5c>
1000004a4: b85f43a0    	ldur	w0, [x29, #-0xc]
1000004a8: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000004ac: 9100c3ff    	add	sp, sp, #0x30
1000004b0: d65f03c0    	ret

00000001000004b4 <_main>:
1000004b4: d10083ff    	sub	sp, sp, #0x20
1000004b8: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000004bc: 910043fd    	add	x29, sp, #0x10
1000004c0: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000004c4: 97ffffe1    	bl	0x100000448 <__Z15test_new_structv>
1000004c8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000004cc: 910083ff    	add	sp, sp, #0x20
1000004d0: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

00000001000004d4 <__stubs>:
1000004d4: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
1000004d8: f9400210    	ldr	x16, [x16]
1000004dc: d61f0200    	br	x16
1000004e0: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
1000004e4: f9400610    	ldr	x16, [x16, #0x8]
1000004e8: d61f0200    	br	x16
