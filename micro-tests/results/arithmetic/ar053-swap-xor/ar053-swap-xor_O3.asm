
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar053-swap-xor/ar053-swap-xor_O3.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z13test_swap_xorRiS_>:
100000360: b9400028    	ldr	w8, [x1]
100000364: b9400009    	ldr	w9, [x0]
100000368: 4a080128    	eor	w8, w9, w8
10000036c: b9000008    	str	w8, [x0]
100000370: b9400029    	ldr	w9, [x1]
100000374: 4a080128    	eor	w8, w9, w8
100000378: b9000028    	str	w8, [x1]
10000037c: b9400009    	ldr	w9, [x0]
100000380: 4a080128    	eor	w8, w9, w8
100000384: b9000008    	str	w8, [x0]
100000388: d65f03c0    	ret

000000010000038c <_main>:
10000038c: 52800140    	mov	w0, #0xa                ; =10
100000390: d65f03c0    	ret
