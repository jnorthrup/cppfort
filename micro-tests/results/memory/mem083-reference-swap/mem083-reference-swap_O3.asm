
/Users/jim/work/cppfort/micro-tests/results/memory/mem083-reference-swap/mem083-reference-swap_O3.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z4swapRiS_>:
100000360: b9400008    	ldr	w8, [x0]
100000364: b9400029    	ldr	w9, [x1]
100000368: b9000009    	str	w9, [x0]
10000036c: b9000028    	str	w8, [x1]
100000370: d65f03c0    	ret

0000000100000374 <_main>:
100000374: 52800140    	mov	w0, #0xa                ; =10
100000378: d65f03c0    	ret
