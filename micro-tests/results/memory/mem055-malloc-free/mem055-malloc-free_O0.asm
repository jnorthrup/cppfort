
/Users/jim/work/cppfort/micro-tests/results/memory/mem055-malloc-free/mem055-malloc-free_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000448 <__Z16test_malloc_freev>:
100000448: d10083ff    	sub	sp, sp, #0x20
10000044c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000450: 910043fd    	add	x29, sp, #0x10
100000454: d2800080    	mov	x0, #0x4                ; =4
100000458: 94000016    	bl	0x1000004b0 <_malloc+0x1000004b0>
10000045c: f90007e0    	str	x0, [sp, #0x8]
100000460: f94007e9    	ldr	x9, [sp, #0x8]
100000464: 52800548    	mov	w8, #0x2a               ; =42
100000468: b9000128    	str	w8, [x9]
10000046c: f94007e8    	ldr	x8, [sp, #0x8]
100000470: b9400108    	ldr	w8, [x8]
100000474: b90007e8    	str	w8, [sp, #0x4]
100000478: f94007e0    	ldr	x0, [sp, #0x8]
10000047c: 94000010    	bl	0x1000004bc <_malloc+0x1000004bc>
100000480: b94007e0    	ldr	w0, [sp, #0x4]
100000484: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000488: 910083ff    	add	sp, sp, #0x20
10000048c: d65f03c0    	ret

0000000100000490 <_main>:
100000490: d10083ff    	sub	sp, sp, #0x20
100000494: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000498: 910043fd    	add	x29, sp, #0x10
10000049c: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000004a0: 97ffffea    	bl	0x100000448 <__Z16test_malloc_freev>
1000004a4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000004a8: 910083ff    	add	sp, sp, #0x20
1000004ac: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

00000001000004b0 <__stubs>:
1000004b0: 90000030    	adrp	x16, 0x100004000 <_malloc+0x100004000>
1000004b4: f9400210    	ldr	x16, [x16]
1000004b8: d61f0200    	br	x16
1000004bc: 90000030    	adrp	x16, 0x100004000 <_malloc+0x100004000>
1000004c0: f9400610    	ldr	x16, [x16, #0x8]
1000004c4: d61f0200    	br	x16
