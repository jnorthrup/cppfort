
/Users/jim/work/cppfort/micro-tests/results/memory/mem072-reference-assignment/mem072-reference-assignment_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z25test_reference_assignmentv>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: 910033e8    	add	x8, sp, #0xc
100000368: 52800149    	mov	w9, #0xa                ; =10
10000036c: b9000fe9    	str	w9, [sp, #0xc]
100000370: f90003e8    	str	x8, [sp]
100000374: f94003e9    	ldr	x9, [sp]
100000378: 52800288    	mov	w8, #0x14               ; =20
10000037c: b9000128    	str	w8, [x9]
100000380: b9400fe0    	ldr	w0, [sp, #0xc]
100000384: 910043ff    	add	sp, sp, #0x10
100000388: d65f03c0    	ret

000000010000038c <_main>:
10000038c: d10083ff    	sub	sp, sp, #0x20
100000390: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000394: 910043fd    	add	x29, sp, #0x10
100000398: b81fc3bf    	stur	wzr, [x29, #-0x4]
10000039c: 97fffff1    	bl	0x100000360 <__Z25test_reference_assignmentv>
1000003a0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003a4: 910083ff    	add	sp, sp, #0x20
1000003a8: d65f03c0    	ret
