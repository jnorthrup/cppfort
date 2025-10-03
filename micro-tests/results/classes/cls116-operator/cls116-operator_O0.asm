
/Users/jim/work/cppfort/micro-tests/results/classes/cls116-operator/cls116-operator_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000448 <_main>:
100000448: d100c3ff    	sub	sp, sp, #0x30
10000044c: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000450: 910083fd    	add	x29, sp, #0x20
100000454: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000458: d10023a0    	sub	x0, x29, #0x8
10000045c: f90003e0    	str	x0, [sp]
100000460: 52800e81    	mov	w1, #0x74               ; =116
100000464: 9400000f    	bl	0x1000004a0 <__ZN4TestC1Ei>
100000468: d10033a0    	sub	x0, x29, #0xc
10000046c: f90007e0    	str	x0, [sp, #0x8]
100000470: 52800021    	mov	w1, #0x1                ; =1
100000474: 9400000b    	bl	0x1000004a0 <__ZN4TestC1Ei>
100000478: f94003e0    	ldr	x0, [sp]
10000047c: f94007e1    	ldr	x1, [sp, #0x8]
100000480: 94000033    	bl	0x10000054c
100000484: aa0003e8    	mov	x8, x0
100000488: 910043e0    	add	x0, sp, #0x10
10000048c: b90013e8    	str	w8, [sp, #0x10]
100000490: 94000032    	bl	0x100000558
100000494: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000498: 9100c3ff    	add	sp, sp, #0x30
10000049c: d65f03c0    	ret

00000001000004a0 <__ZN4TestC1Ei>:
1000004a0: d100c3ff    	sub	sp, sp, #0x30
1000004a4: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000004a8: 910083fd    	add	x29, sp, #0x20
1000004ac: f81f83a0    	stur	x0, [x29, #-0x8]
1000004b0: b81f43a1    	stur	w1, [x29, #-0xc]
1000004b4: f85f83a0    	ldur	x0, [x29, #-0x8]
1000004b8: f90007e0    	str	x0, [sp, #0x8]
1000004bc: b85f43a1    	ldur	w1, [x29, #-0xc]
1000004c0: 9400001b    	bl	0x10000052c <__ZN4TestC2Ei>
1000004c4: f94007e0    	ldr	x0, [sp, #0x8]
1000004c8: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000004cc: 9100c3ff    	add	sp, sp, #0x30
1000004d0: d65f03c0    	ret

00000001000004d4 <__ZN4TestplERKS_>:
1000004d4: d100c3ff    	sub	sp, sp, #0x30
1000004d8: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000004dc: 910083fd    	add	x29, sp, #0x20
1000004e0: f9000be0    	str	x0, [sp, #0x10]
1000004e4: f90007e1    	str	x1, [sp, #0x8]
1000004e8: f9400be8    	ldr	x8, [sp, #0x10]
1000004ec: b9400108    	ldr	w8, [x8]
1000004f0: f94007e9    	ldr	x9, [sp, #0x8]
1000004f4: b9400129    	ldr	w9, [x9]
1000004f8: 0b090101    	add	w1, w8, w9
1000004fc: d10013a0    	sub	x0, x29, #0x4
100000500: 97ffffe8    	bl	0x1000004a0 <__ZN4TestC1Ei>
100000504: b85fc3a0    	ldur	w0, [x29, #-0x4]
100000508: a9427bfd    	ldp	x29, x30, [sp, #0x20]
10000050c: 9100c3ff    	add	sp, sp, #0x30
100000510: d65f03c0    	ret

0000000100000514 <__ZN4Test3getEv>:
100000514: d10043ff    	sub	sp, sp, #0x10
100000518: f90007e0    	str	x0, [sp, #0x8]
10000051c: f94007e8    	ldr	x8, [sp, #0x8]
100000520: b9400100    	ldr	w0, [x8]
100000524: 910043ff    	add	sp, sp, #0x10
100000528: d65f03c0    	ret

000000010000052c <__ZN4TestC2Ei>:
10000052c: d10043ff    	sub	sp, sp, #0x10
100000530: f90007e0    	str	x0, [sp, #0x8]
100000534: b90007e1    	str	w1, [sp, #0x4]
100000538: f94007e0    	ldr	x0, [sp, #0x8]
10000053c: b94007e8    	ldr	w8, [sp, #0x4]
100000540: b9000008    	str	w8, [x0]
100000544: 910043ff    	add	sp, sp, #0x10
100000548: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

000000010000054c <__stubs>:
10000054c: 90000030    	adrp	x16, 0x100004000
100000550: f9400210    	ldr	x16, [x16]
100000554: d61f0200    	br	x16
100000558: 90000030    	adrp	x16, 0x100004000
10000055c: f9400610    	ldr	x16, [x16, #0x8]
100000560: d61f0200    	br	x16
