
/Users/jim/work/cppfort/micro-tests/results/memory/mem078-perfect-forwarding/mem078-perfect-forwarding_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000448 <__Z7processRi>:
100000448: d10043ff    	sub	sp, sp, #0x10
10000044c: f90007e0    	str	x0, [sp, #0x8]
100000450: f94007e8    	ldr	x8, [sp, #0x8]
100000454: b9400100    	ldr	w0, [x8]
100000458: 910043ff    	add	sp, sp, #0x10
10000045c: d65f03c0    	ret

0000000100000460 <_main>:
100000460: d10083ff    	sub	sp, sp, #0x20
100000464: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000468: 910043fd    	add	x29, sp, #0x10
10000046c: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000470: 910023e0    	add	x0, sp, #0x8
100000474: 52800548    	mov	w8, #0x2a               ; =42
100000478: b9000be8    	str	w8, [sp, #0x8]
10000047c: 9400000d    	bl	0x1000004b0
100000480: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000484: 910083ff    	add	sp, sp, #0x20
100000488: d65f03c0    	ret

000000010000048c <__Z12forward_callIRiEiOT_>:
10000048c: d10083ff    	sub	sp, sp, #0x20
100000490: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000494: 910043fd    	add	x29, sp, #0x10
100000498: f90007e0    	str	x0, [sp, #0x8]
10000049c: f94007e0    	ldr	x0, [sp, #0x8]
1000004a0: 97ffffea    	bl	0x100000448 <__Z7processRi>
1000004a4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000004a8: 910083ff    	add	sp, sp, #0x20
1000004ac: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

00000001000004b0 <__stubs>:
1000004b0: 90000030    	adrp	x16, 0x100004000
1000004b4: f9400210    	ldr	x16, [x16]
1000004b8: d61f0200    	br	x16
