
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar064-signed-unsigned-compare/ar064-signed-unsigned-compare_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z23test_mixed_sign_compareij>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fe0    	str	w0, [sp, #0xc]
100000368: b9000be1    	str	w1, [sp, #0x8]
10000036c: b9400fe8    	ldr	w8, [sp, #0xc]
100000370: b9400be9    	ldr	w9, [sp, #0x8]
100000374: 6b090108    	subs	w8, w8, w9
100000378: 1a9fa7e0    	cset	w0, lt
10000037c: 910043ff    	add	sp, sp, #0x10
100000380: d65f03c0    	ret

0000000100000384 <_main>:
100000384: d10083ff    	sub	sp, sp, #0x20
100000388: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000038c: 910043fd    	add	x29, sp, #0x10
100000390: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000394: 12800080    	mov	w0, #-0x5               ; =-5
100000398: 52800141    	mov	w1, #0xa                ; =10
10000039c: 97fffff1    	bl	0x100000360 <__Z23test_mixed_sign_compareij>
1000003a0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003a4: 910083ff    	add	sp, sp, #0x20
1000003a8: d65f03c0    	ret
