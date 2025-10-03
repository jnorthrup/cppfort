
/Users/jim/work/cppfort/micro-tests/results/classes/cls016-basic/cls016-basic_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000448 <_main>:
100000448: d10083ff    	sub	sp, sp, #0x20
10000044c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000450: 910043fd    	add	x29, sp, #0x10
100000454: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000458: 910023e0    	add	x0, sp, #0x8
10000045c: f90003e0    	str	x0, [sp]
100000460: 52800201    	mov	w1, #0x10               ; =16
100000464: 94000006    	bl	0x10000047c <__ZN4TestC1Ei>
100000468: f94003e0    	ldr	x0, [sp]
10000046c: 9400001f    	bl	0x1000004e8
100000470: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000474: 910083ff    	add	sp, sp, #0x20
100000478: d65f03c0    	ret

000000010000047c <__ZN4TestC1Ei>:
10000047c: d100c3ff    	sub	sp, sp, #0x30
100000480: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000484: 910083fd    	add	x29, sp, #0x20
100000488: f81f83a0    	stur	x0, [x29, #-0x8]
10000048c: b81f43a1    	stur	w1, [x29, #-0xc]
100000490: f85f83a0    	ldur	x0, [x29, #-0x8]
100000494: f90007e0    	str	x0, [sp, #0x8]
100000498: b85f43a1    	ldur	w1, [x29, #-0xc]
10000049c: 9400000b    	bl	0x1000004c8 <__ZN4TestC2Ei>
1000004a0: f94007e0    	ldr	x0, [sp, #0x8]
1000004a4: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000004a8: 9100c3ff    	add	sp, sp, #0x30
1000004ac: d65f03c0    	ret

00000001000004b0 <__ZN4Test3getEv>:
1000004b0: d10043ff    	sub	sp, sp, #0x10
1000004b4: f90007e0    	str	x0, [sp, #0x8]
1000004b8: f94007e8    	ldr	x8, [sp, #0x8]
1000004bc: b9400100    	ldr	w0, [x8]
1000004c0: 910043ff    	add	sp, sp, #0x10
1000004c4: d65f03c0    	ret

00000001000004c8 <__ZN4TestC2Ei>:
1000004c8: d10043ff    	sub	sp, sp, #0x10
1000004cc: f90007e0    	str	x0, [sp, #0x8]
1000004d0: b90007e1    	str	w1, [sp, #0x4]
1000004d4: f94007e0    	ldr	x0, [sp, #0x8]
1000004d8: b94007e8    	ldr	w8, [sp, #0x4]
1000004dc: b9000008    	str	w8, [x0]
1000004e0: 910043ff    	add	sp, sp, #0x10
1000004e4: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

00000001000004e8 <__stubs>:
1000004e8: 90000030    	adrp	x16, 0x100004000
1000004ec: f9400210    	ldr	x16, [x16]
1000004f0: d61f0200    	br	x16
