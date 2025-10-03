
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf010-early-return/cf010-early-return_O2.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z17test_early_returni>:
100000360: 52800c88    	mov	w8, #0x64               ; =100
100000364: 7101901f    	cmp	w0, #0x64
100000368: 1a883008    	csel	w8, w0, w8, lo
10000036c: 7100001f    	cmp	w0, #0x0
100000370: 1a88b3e0    	csel	w0, wzr, w8, lt
100000374: d65f03c0    	ret

0000000100000378 <_main>:
100000378: 52800640    	mov	w0, #0x32               ; =50
10000037c: d65f03c0    	ret
