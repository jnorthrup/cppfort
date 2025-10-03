
/Users/jim/work/cppfort/micro-tests/results/memory/mem052-memory-leak/mem052-memory-leak_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000448 <__Z16test_memory_leakv>:
100000448: d10083ff    	sub	sp, sp, #0x20
10000044c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000450: 910043fd    	add	x29, sp, #0x10
100000454: d2800080    	mov	x0, #0x4                ; =4
100000458: 94000011    	bl	0x10000049c <__Znwm+0x10000049c>
10000045c: 52800548    	mov	w8, #0x2a               ; =42
100000460: b9000008    	str	w8, [x0]
100000464: f90007e0    	str	x0, [sp, #0x8]
100000468: f94007e8    	ldr	x8, [sp, #0x8]
10000046c: b9400100    	ldr	w0, [x8]
100000470: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000474: 910083ff    	add	sp, sp, #0x20
100000478: d65f03c0    	ret

000000010000047c <_main>:
10000047c: d10083ff    	sub	sp, sp, #0x20
100000480: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000484: 910043fd    	add	x29, sp, #0x10
100000488: b81fc3bf    	stur	wzr, [x29, #-0x4]
10000048c: 97ffffef    	bl	0x100000448 <__Z16test_memory_leakv>
100000490: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000494: 910083ff    	add	sp, sp, #0x20
100000498: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

000000010000049c <__stubs>:
10000049c: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
1000004a0: f9400210    	ldr	x16, [x16]
1000004a4: d61f0200    	br	x16
