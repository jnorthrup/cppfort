
/Users/jim/work/cppfort/micro-tests/results/memory/mem057-realloc/mem057-realloc_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000448 <__Z12test_reallocv>:
100000448: d10083ff    	sub	sp, sp, #0x20
10000044c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000450: 910043fd    	add	x29, sp, #0x10
100000454: d2800280    	mov	x0, #0x14               ; =20
100000458: 9400001a    	bl	0x1000004c0 <_realloc+0x1000004c0>
10000045c: f90007e0    	str	x0, [sp, #0x8]
100000460: f94007e9    	ldr	x9, [sp, #0x8]
100000464: 52800548    	mov	w8, #0x2a               ; =42
100000468: b9000928    	str	w8, [x9, #0x8]
10000046c: f94007e0    	ldr	x0, [sp, #0x8]
100000470: d2800501    	mov	x1, #0x28               ; =40
100000474: 94000016    	bl	0x1000004cc <_realloc+0x1000004cc>
100000478: f90007e0    	str	x0, [sp, #0x8]
10000047c: f94007e8    	ldr	x8, [sp, #0x8]
100000480: b9400908    	ldr	w8, [x8, #0x8]
100000484: b90007e8    	str	w8, [sp, #0x4]
100000488: f94007e0    	ldr	x0, [sp, #0x8]
10000048c: 94000013    	bl	0x1000004d8 <_realloc+0x1000004d8>
100000490: b94007e0    	ldr	w0, [sp, #0x4]
100000494: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000498: 910083ff    	add	sp, sp, #0x20
10000049c: d65f03c0    	ret

00000001000004a0 <_main>:
1000004a0: d10083ff    	sub	sp, sp, #0x20
1000004a4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000004a8: 910043fd    	add	x29, sp, #0x10
1000004ac: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000004b0: 97ffffe6    	bl	0x100000448 <__Z12test_reallocv>
1000004b4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000004b8: 910083ff    	add	sp, sp, #0x20
1000004bc: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

00000001000004c0 <__stubs>:
1000004c0: 90000030    	adrp	x16, 0x100004000 <_realloc+0x100004000>
1000004c4: f9400210    	ldr	x16, [x16]
1000004c8: d61f0200    	br	x16
1000004cc: 90000030    	adrp	x16, 0x100004000 <_realloc+0x100004000>
1000004d0: f9400610    	ldr	x16, [x16, #0x8]
1000004d4: d61f0200    	br	x16
1000004d8: 90000030    	adrp	x16, 0x100004000 <_realloc+0x100004000>
1000004dc: f9400a10    	ldr	x16, [x16, #0x10]
1000004e0: d61f0200    	br	x16
