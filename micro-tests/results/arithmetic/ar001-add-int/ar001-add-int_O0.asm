
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar001-add-int/ar001-add-int_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z8test_addii>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fe0    	str	w0, [sp, #0xc]
100000368: b9000be1    	str	w1, [sp, #0x8]
10000036c: b9400fe8    	ldr	w8, [sp, #0xc]
100000370: b9400be9    	ldr	w9, [sp, #0x8]
100000374: 0b090100    	add	w0, w8, w9
100000378: 910043ff    	add	sp, sp, #0x10
10000037c: d65f03c0    	ret

0000000100000380 <_main>:
100000380: d10083ff    	sub	sp, sp, #0x20
100000384: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000388: 910043fd    	add	x29, sp, #0x10
10000038c: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000390: 528000a0    	mov	w0, #0x5                ; =5
100000394: 52800061    	mov	w1, #0x3                ; =3
100000398: 97fffff2    	bl	0x100000360 <__Z8test_addii>
10000039c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003a0: 910083ff    	add	sp, sp, #0x20
1000003a4: d65f03c0    	ret
