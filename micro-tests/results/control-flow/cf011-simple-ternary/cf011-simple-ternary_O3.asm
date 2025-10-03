
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf011-simple-ternary/cf011-simple-ternary_O3.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z12test_ternaryi>:
100000360: 7100041f    	cmp	w0, #0x1
100000364: 12800008    	mov	w8, #-0x1               ; =-1
100000368: 5a88b500    	cneg	w0, w8, ge
10000036c: d65f03c0    	ret

0000000100000370 <_main>:
100000370: 52800020    	mov	w0, #0x1                ; =1
100000374: d65f03c0    	ret
