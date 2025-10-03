
/Users/jim/work/cppfort/micro-tests/results/memory/mem041-restrict-pointer/mem041-restrict-pointer_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z13test_restrictPiS_>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: f90007e0    	str	x0, [sp, #0x8]
100000368: f90003e1    	str	x1, [sp]
10000036c: f94007e9    	ldr	x9, [sp, #0x8]
100000370: 52800148    	mov	w8, #0xa                ; =10
100000374: b9000128    	str	w8, [x9]
100000378: f94003e9    	ldr	x9, [sp]
10000037c: 52800288    	mov	w8, #0x14               ; =20
100000380: b9000128    	str	w8, [x9]
100000384: f94007e8    	ldr	x8, [sp, #0x8]
100000388: b9400100    	ldr	w0, [x8]
10000038c: 910043ff    	add	sp, sp, #0x10
100000390: d65f03c0    	ret

0000000100000394 <_main>:
100000394: d10083ff    	sub	sp, sp, #0x20
100000398: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000039c: 910043fd    	add	x29, sp, #0x10
1000003a0: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003a4: 910023e0    	add	x0, sp, #0x8
1000003a8: b9000bff    	str	wzr, [sp, #0x8]
1000003ac: 910013e1    	add	x1, sp, #0x4
1000003b0: b90007ff    	str	wzr, [sp, #0x4]
1000003b4: 97ffffeb    	bl	0x100000360 <__Z13test_restrictPiS_>
1000003b8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003bc: 910083ff    	add	sp, sp, #0x20
1000003c0: d65f03c0    	ret
