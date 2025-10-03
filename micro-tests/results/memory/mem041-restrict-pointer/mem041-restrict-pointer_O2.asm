
/Users/jim/work/cppfort/micro-tests/results/memory/mem041-restrict-pointer/mem041-restrict-pointer_O2.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z13test_restrictPiS_>:
100000360: 52800148    	mov	w8, #0xa                ; =10
100000364: b9000008    	str	w8, [x0]
100000368: 52800288    	mov	w8, #0x14               ; =20
10000036c: b9000028    	str	w8, [x1]
100000370: 52800140    	mov	w0, #0xa                ; =10
100000374: d65f03c0    	ret

0000000100000378 <_main>:
100000378: 52800140    	mov	w0, #0xa                ; =10
10000037c: d65f03c0    	ret
