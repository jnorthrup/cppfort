
/Users/jim/work/cppfort/micro-tests/results/classes/cls052-inherit/cls052-inherit_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000448 <_main>:
100000448: d100c3ff    	sub	sp, sp, #0x30
10000044c: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000450: 910083fd    	add	x29, sp, #0x20
100000454: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000458: 910043e0    	add	x0, sp, #0x10
10000045c: f90007e0    	str	x0, [sp, #0x8]
100000460: 94000006    	bl	0x100000478 <__ZN7DerivedC1Ev>
100000464: f94007e0    	ldr	x0, [sp, #0x8]
100000468: 9400000f    	bl	0x1000004a4 <__ZN7Derived3getEv>
10000046c: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000470: 9100c3ff    	add	sp, sp, #0x30
100000474: d65f03c0    	ret

0000000100000478 <__ZN7DerivedC1Ev>:
100000478: d10083ff    	sub	sp, sp, #0x20
10000047c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000480: 910043fd    	add	x29, sp, #0x10
100000484: f90007e0    	str	x0, [sp, #0x8]
100000488: f94007e0    	ldr	x0, [sp, #0x8]
10000048c: f90003e0    	str	x0, [sp]
100000490: 9400000a    	bl	0x1000004b8 <__ZN7DerivedC2Ev>
100000494: f94003e0    	ldr	x0, [sp]
100000498: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000049c: 910083ff    	add	sp, sp, #0x20
1000004a0: d65f03c0    	ret

00000001000004a4 <__ZN7Derived3getEv>:
1000004a4: d10043ff    	sub	sp, sp, #0x10
1000004a8: f90007e0    	str	x0, [sp, #0x8]
1000004ac: 528006a0    	mov	w0, #0x35               ; =53
1000004b0: 910043ff    	add	sp, sp, #0x10
1000004b4: d65f03c0    	ret

00000001000004b8 <__ZN7DerivedC2Ev>:
1000004b8: d10083ff    	sub	sp, sp, #0x20
1000004bc: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000004c0: 910043fd    	add	x29, sp, #0x10
1000004c4: f90007e0    	str	x0, [sp, #0x8]
1000004c8: f94007e0    	ldr	x0, [sp, #0x8]
1000004cc: f90003e0    	str	x0, [sp]
1000004d0: 94000009    	bl	0x1000004f4 <__ZN4BaseC2Ev>
1000004d4: f94003e0    	ldr	x0, [sp]
1000004d8: 90000028    	adrp	x8, 0x100004000 <__ZTV7Derived>
1000004dc: 91000108    	add	x8, x8, #0x0
1000004e0: 91004108    	add	x8, x8, #0x10
1000004e4: f9000008    	str	x8, [x0]
1000004e8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000004ec: 910083ff    	add	sp, sp, #0x20
1000004f0: d65f03c0    	ret

00000001000004f4 <__ZN4BaseC2Ev>:
1000004f4: d10043ff    	sub	sp, sp, #0x10
1000004f8: f90007e0    	str	x0, [sp, #0x8]
1000004fc: f94007e0    	ldr	x0, [sp, #0x8]
100000500: 90000028    	adrp	x8, 0x100004000 <__ZTV7Derived>
100000504: 91010108    	add	x8, x8, #0x40
100000508: 91004108    	add	x8, x8, #0x10
10000050c: f9000008    	str	x8, [x0]
100000510: 910043ff    	add	sp, sp, #0x10
100000514: d65f03c0    	ret

0000000100000518 <__ZN4Base3getEv>:
100000518: d10043ff    	sub	sp, sp, #0x10
10000051c: f90007e0    	str	x0, [sp, #0x8]
100000520: 52800680    	mov	w0, #0x34               ; =52
100000524: 910043ff    	add	sp, sp, #0x10
100000528: d65f03c0    	ret
