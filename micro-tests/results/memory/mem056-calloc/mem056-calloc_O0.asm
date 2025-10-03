
/Users/jim/work/cppfort/micro-tests/results/memory/mem056-calloc/mem056-calloc_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000448 <__Z11test_callocv>:
100000448: d10083ff    	sub	sp, sp, #0x20
10000044c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000450: 910043fd    	add	x29, sp, #0x10
100000454: d28000a0    	mov	x0, #0x5                ; =5
100000458: d2800081    	mov	x1, #0x4                ; =4
10000045c: 94000016    	bl	0x1000004b4 <_free+0x1000004b4>
100000460: f90007e0    	str	x0, [sp, #0x8]
100000464: f94007e9    	ldr	x9, [sp, #0x8]
100000468: 52800548    	mov	w8, #0x2a               ; =42
10000046c: b9000928    	str	w8, [x9, #0x8]
100000470: f94007e8    	ldr	x8, [sp, #0x8]
100000474: b9400908    	ldr	w8, [x8, #0x8]
100000478: b90007e8    	str	w8, [sp, #0x4]
10000047c: f94007e0    	ldr	x0, [sp, #0x8]
100000480: 94000010    	bl	0x1000004c0 <_free+0x1000004c0>
100000484: b94007e0    	ldr	w0, [sp, #0x4]
100000488: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000048c: 910083ff    	add	sp, sp, #0x20
100000490: d65f03c0    	ret

0000000100000494 <_main>:
100000494: d10083ff    	sub	sp, sp, #0x20
100000498: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000049c: 910043fd    	add	x29, sp, #0x10
1000004a0: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000004a4: 97ffffe9    	bl	0x100000448 <__Z11test_callocv>
1000004a8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000004ac: 910083ff    	add	sp, sp, #0x20
1000004b0: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

00000001000004b4 <__stubs>:
1000004b4: 90000030    	adrp	x16, 0x100004000 <_free+0x100004000>
1000004b8: f9400210    	ldr	x16, [x16]
1000004bc: d61f0200    	br	x16
1000004c0: 90000030    	adrp	x16, 0x100004000 <_free+0x100004000>
1000004c4: f9400610    	ldr	x16, [x16, #0x8]
1000004c8: d61f0200    	br	x16
