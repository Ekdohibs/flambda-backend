(******************************************************************************
 *                                  OxCaml                                    *
 * -------------------------------------------------------------------------- *
 *                               MIT License                                  *
 *                                                                            *
 * Copyright (c) 2026 Jane Street Group LLC                                   *
 * opensource-contacts@janestreet.com                                         *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included    *
 * in all copies or substantial portions of the Software.                     *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER *
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    *
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        *
 * DEALINGS IN THE SOFTWARE.                                                  *
 ******************************************************************************)

(* Proofs of the statements of FlambdaCompareAlphaSpec.v. Nothing here
   needs review.

   Symmetry: the comparison of two terms under pairs of names is that of
   the terms the other way round under the pairs turned round ([flip]).
   Transitivity: two comparisons compose, under maps of names that
   compose ([composes]: a property of what the maps match, kept when a
   pair is put on each). A context keeps what the comparison accepts
   ([plug_eqb]); the rules that rename a binder are accepted
   (FlambdaNominalProofs.v). *)

From Stdlib Require Import ZArith List Bool Arith.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaPrimitiveSpec FlambdaPrimitiveProofs
  FlambdaBindersSpec FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaContextSpec FlambdaNominalSpec FlambdaNominalProofs
  FlambdaAlphaSpec FlambdaAlphaProofs
  FlambdaCompareSpec FlambdaCompareProofs FlambdaCompareAlphaSpec.
Import ListNotations.

Lemma const_eqb_eq (c c' : const) : const_eqb c c' = true -> c = c'.
Proof.
  unfold const_eqb. destruct (const_eq_dec c c'); [ auto | discriminate ].
Qed.

Lemma call_kind_eqb_refl (kind : call_kind) : call_kind_eqb kind kind = true.
Proof. destruct kind; simpl; [ apply Pos.eqb_refl | reflexivity ]. Qed.

(* ======================================================================== *)
(* Symmetry                                                                 *)
(* ======================================================================== *)

Definition flip (m : name_map) : name_map :=
  map (fun pair => (snd pair, fst pair)) m.

Lemma names_match_flip (m : name_map) (a a' : positive) :
  names_match (flip m) a' a = names_match m a a'.
Proof.
  induction m as [ | [b b'] m IH]; simpl; [ apply Pos.eqb_sym | ].
  rewrite IH. destruct (Pos.eqb a b), (Pos.eqb a' b'); reflexivity.
Qed.

Lemma bind_names_flip (xs : list positive) :
  ∀ (xs' : list positive) (m : name_map),
    bind_names xs' xs (flip m) = option_map flip (bind_names xs xs' m).
Proof.
  induction xs as [ | x xs IH]; intros [ | x' xs'] m; simpl;
    try reflexivity.
  exact (IH xs' ((x, x') :: m)).
Qed.

Lemma flip_combine (xs : list positive) :
  ∀ xs' : list positive, flip (combine xs xs') = combine xs' xs.
Proof.
  induction xs as [ | x xs IH]; intros [ | x' xs']; simpl; try reflexivity.
  rewrite <- IH. reflexivity.
Qed.

Lemma combine_names_flip (xs xs' : list positive) (m : name_map) :
  combine_names xs' xs (flip m) = option_map flip (combine_names xs xs' m).
Proof.
  unfold combine_names. rewrite (Nat.eqb_sym (length xs') (length xs)).
  destruct (Nat.eqb (length xs) (length xs')); [ | reflexivity ].
  cbn [option_map]. unfold flip at 2. rewrite map_app.
  fold (flip (combine xs xs')). fold (flip m). rewrite flip_combine.
  reflexivity.
Qed.

Lemma forall2b_flip {A B : Type} (f : A -> B -> bool) (g : B -> A -> bool)
    (l : list A) :
  (∀ (a : A) (b : B), f a b = true -> g b a = true) ->
  ∀ l' : list B, forall2b f l l' = true -> forall2b g l' l = true.
Proof.
  intros Hfg. induction l as [ | a l IH]; intros [ | b l']; simpl;
    try discriminate; [ reflexivity | ].
  intros Heq. apply andb_true_iff in Heq as [Hab Hl].
  rewrite (Hfg a b Hab), (IH l' Hl). reflexivity.
Qed.

Lemma simple_eqb_flip (vm : name_map) (s s' : simple) :
  simple_eqb vm s s' = true -> simple_eqb (flip vm) s' s = true.
Proof.
  destruct s as [x | sym | c], s' as [x' | sym' | c']; simpl;
    try discriminate.
  - rewrite names_match_flip. auto.
  - rewrite Pos.eqb_sym. auto.
  - intros Heq. apply const_eqb_eq in Heq. subst c'. apply const_eqb_refl.
Qed.

Lemma simples_eqb_flip (vm : name_map) (args args' : list simple) :
  simples_eqb vm args args' = true -> simples_eqb (flip vm) args' args = true.
Proof. apply forall2b_flip. intros s s'. apply simple_eqb_flip. Qed.

Lemma value_slots_eqb_flip (vm : name_map)
    (slots slots' : list (value_slot * simple)) :
  value_slots_eqb vm slots slots' = true ->
  value_slots_eqb (flip vm) slots' slots = true.
Proof.
  apply forall2b_flip. intros slot slot' Heq.
  apply andb_true_iff in Heq as [Hslot Hs].
  rewrite Pos.eqb_sym, Hslot. apply simple_eqb_flip, Hs.
Qed.

Lemma named_eqb_flip (vm : name_map) (d d' : named) :
  named_eqb vm d d' = true -> named_eqb (flip vm) d' d = true.
Proof.
  destruct d
    as [s | p args | code slots | fslot decls slots | slot closure
       | pfrom slot closure | move_from move_to closure],
    d'
    as [s' | p' args' | code' slots' | fslot' decls' slots' | slot' closure'
       | pfrom' slot' closure' | move_from' move_to' closure'];
    simpl; try discriminate; intros Heq.
  - apply simple_eqb_flip, Heq.
  - apply andb_true_iff in Heq as [Hp Hargs]. apply prim_eqb_sound in Hp.
    subst p'. rewrite prim_eqb_refl. apply simples_eqb_flip, Hargs.
  - apply andb_true_iff in Heq as [Hcode Hslots]. rewrite Pos.eqb_sym, Hcode.
    apply value_slots_eqb_flip, Hslots.
  - apply andb_true_iff in Heq as [Heq Hslots].
    apply andb_true_iff in Heq as [Hfslot Hdecls].
    apply Pos.eqb_eq in Hfslot. apply names_eqb_eq in Hdecls.
    subst fslot' decls'. rewrite Pos.eqb_refl, names_eqb_refl.
    apply value_slots_eqb_flip, Hslots.
  - apply andb_true_iff in Heq as [Hslot Hclosure].
    rewrite Pos.eqb_sym, Hslot. apply simple_eqb_flip, Hclosure.
  - apply andb_true_iff in Heq as [Heq Hclosure].
    apply andb_true_iff in Heq as [Hfrom Hslot].
    rewrite Pos.eqb_sym, Hfrom, Pos.eqb_sym, Hslot.
    apply simple_eqb_flip, Hclosure.
  - apply andb_true_iff in Heq as [Heq Hclosure].
    apply andb_true_iff in Heq as [Hfrom Hto].
    rewrite Pos.eqb_sym, Hfrom, Pos.eqb_sym, Hto.
    apply simple_eqb_flip, Hclosure.
Qed.

Lemma static_const_eqb_flip (vm : name_map) (const const' : static_const) :
  static_const_eqb vm const const' = true ->
  static_const_eqb (flip vm) const' const = true.
Proof.
  destruct const
    as [code slots | fslot syms decls slots | tag fields | s | bytes
        | elements],
    const'
    as [code' slots' | fslot' syms' decls' slots' | tag' fields' | s'
       | bytes' | elements'];
    simpl; try discriminate; intros Heq.
  - apply andb_true_iff in Heq as [Hcode Hslots]. rewrite Pos.eqb_sym, Hcode.
    apply value_slots_eqb_flip, Hslots.
  - apply andb_true_iff in Heq as [Heq Hslots].
    apply andb_true_iff in Heq as [Heq Hdecls].
    apply andb_true_iff in Heq as [Hfslot Hsyms].
    apply Pos.eqb_eq in Hfslot. apply names_eqb_eq in Hsyms, Hdecls.
    subst fslot' syms' decls'. rewrite Pos.eqb_refl, !names_eqb_refl.
    apply value_slots_eqb_flip, Hslots.
  - apply andb_true_iff in Heq as [Htag Hfields]. rewrite Z.eqb_sym, Htag.
    apply simples_eqb_flip, Hfields.
  - apply simple_eqb_flip, Heq.
  - apply (forall2b_flip Z.eqb Z.eqb bytes); [ | exact Heq ].
    intros z z' Hz. rewrite Z.eqb_sym. exact Hz.
  - apply simples_eqb_flip, Heq.
Qed.

Lemma apply_cont_eqb_flip (vm km : name_map) (ac ac' : apply_cont) :
  apply_cont_eqb vm km ac ac' = true ->
  apply_cont_eqb (flip vm) (flip km) ac' ac = true.
Proof.
  unfold apply_cont_eqb. intros Heq.
  apply andb_true_iff in Heq as [Heq Htrap].
  apply andb_true_iff in Heq as [Hk Hargs].
  rewrite names_match_flip, Hk, (simples_eqb_flip _ _ _ Hargs). simpl.
  destruct (ac_trap_action ac) as [[k | k kind] | ],
    (ac_trap_action ac') as [[k' | k' kind'] | ];
    simpl in Htrap |- *; try discriminate; try reflexivity.
  - rewrite names_match_flip. exact Htrap.
  - apply andb_true_iff in Htrap as [Hk' Hkind].
    rewrite names_match_flip, Hk'. simpl.
    destruct kind as [[ | | ] | ], kind' as [[ | | ] | ];
      simpl in Hkind |- *; try discriminate; reflexivity.
Qed.

Lemma apply_eqb_flip (vm km : name_map) (ap ap' : apply_expr) :
  apply_eqb vm km ap ap' = true ->
  apply_eqb (flip vm) (flip km) ap' ap = true.
Proof.
  unfold apply_eqb. intros Heq.
  apply andb_true_iff in Heq as [Heq Hreturn_arity].
  apply andb_true_iff in Heq as [Heq Hargs_arity].
  apply andb_true_iff in Heq as [Heq Hkinds].
  apply andb_true_iff in Heq as [Heq Hextra].
  apply andb_true_iff in Heq as [Heq Hexn].
  apply andb_true_iff in Heq as [Heq Hreturn].
  apply andb_true_iff in Heq as [Heq Hargs].
  apply andb_true_iff in Heq as [Hcallee Hkind].
  apply call_kind_eqb_eq in Hkind. apply kinds_eqb_eq in Hkinds.
  apply arity_eqb_eq in Hargs_arity, Hreturn_arity.
  rewrite !names_match_flip, Hexn, Hreturn, (simple_eqb_flip _ _ _ Hcallee),
    (simples_eqb_flip _ _ _ Hargs), (simples_eqb_flip _ _ _ Hextra),
    <- Hkind, call_kind_eqb_refl, <- Hkinds, kinds_eqb_refl,
    <- Hargs_arity, <- Hreturn_arity, !arity_eqb_refl.
  reflexivity.
Qed.

Fixpoint expr_eqb_flip (e : expr) :
  ∀ (vm km : name_map) (e' : expr),
    expr_eqb vm km e e' = true -> expr_eqb (flip vm) (flip km) e' e = true
with handler_eqb_flip (h : continuation_handler) :
  ∀ (vm km : name_map) (h' : continuation_handler),
    handler_eqb vm km h h' = true ->
    handler_eqb (flip vm) (flip km) h' h = true
with binding_eqb_flip (binding : static_binding) :
  ∀ (vm : name_map) (binding' : static_binding),
    binding_eqb vm binding binding' = true ->
    binding_eqb (flip vm) binding' binding = true
with code_eqb_flip (c : code) :
  ∀ c' : code, code_eqb c c' = true -> code_eqb c' c = true.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros vm km
        [ x' kind' d' body' | bindings' body' | k' h' body' | handlers' body'
        | ap' | ac' | scrutinee' arms' | ];
      simpl; try discriminate; intros Heq.
    + (* Let *)
      apply andb_true_iff in Heq as [Hd Hbody].
      apply andb_true_iff in Hd as [Hkind Hd].
      apply kind_eqb_eq in Hkind. subst kind'.
      rewrite kind_eqb_refl, (named_eqb_flip vm d d' Hd).
      exact (expr_eqb_flip body _ _ _ Hbody).
    + (* Let_static *)
      apply andb_true_iff in Heq as [Hbindings Hbody].
      rewrite (expr_eqb_flip body _ _ _ Hbody), andb_true_r.
      clear Hbody. revert bindings' Hbindings.
      induction bindings as [ | binding bindings IH];
        intros [ | binding' bindings']; simpl; try discriminate;
        [ reflexivity | ].
      intros Heq. apply andb_true_iff in Heq as [Hfirst Hrest].
      rewrite (binding_eqb_flip binding _ _ Hfirst), (IH _ Hrest).
      reflexivity.
    + (* Let_cont_non_recursive *)
      apply andb_true_iff in Heq as [Hh Hbody].
      rewrite (handler_eqb_flip h _ _ _ Hh).
      exact (expr_eqb_flip body _ _ _ Hbody).
    + (* Let_cont_recursive *)
      rewrite combine_names_flip. revert Heq.
      destruct (combine_names (map fst handlers) (map fst handlers') km)
        as [km' | ]; [ | discriminate ].
      intros Heq. simpl. apply andb_true_iff in Heq as [Hhandlers Hbody].
      rewrite (expr_eqb_flip body _ _ _ Hbody), andb_true_r.
      clear Hbody. revert handlers' Hhandlers.
      induction handlers as [ | [k h] handlers IH];
        intros [ | [k' h'] handlers']; simpl; try discriminate;
        [ reflexivity | ].
      intros Heq. apply andb_true_iff in Heq as [Hfirst Hrest].
      rewrite (handler_eqb_flip h _ _ _ Hfirst), (IH _ Hrest). reflexivity.
    + (* Apply *)
      apply apply_eqb_flip, Heq.
    + (* Apply_cont *)
      apply apply_cont_eqb_flip, Heq.
    + (* Switch *)
      apply andb_true_iff in Heq as [Hscrutinee Harms].
      rewrite (simple_eqb_flip _ _ _ Hscrutinee).
      revert Harms. apply forall2b_flip. intros arm arm' Harm.
      apply andb_true_iff in Harm as [Hz Hac].
      rewrite Z.eqb_sym, Hz. apply apply_cont_eqb_flip, Hac.
    + (* Invalid *)
      reflexivity.
  - destruct h as [params metadata handler].
    intros vm km [params' metadata' handler']. simpl.
    rewrite bind_names_flip.
    destruct (bind_names params params' vm) as [vm' | ]; [ | discriminate ].
    simpl. intros Heq. apply andb_true_iff in Heq as [Hmetadata Heq].
    apply handler_metadata_eqb_eq in Hmetadata. subst metadata'.
    rewrite handler_metadata_eqb_refl.
    exact (expr_eqb_flip handler _ _ _ Heq).
  - destruct binding as [id c | id | sym const];
      intros vm [id' c' | id' | sym' const']; simpl; try discriminate;
      intros Heq.
    + apply andb_true_iff in Heq as [Hid Hc]. rewrite Pos.eqb_sym, Hid.
      exact (code_eqb_flip c c' Hc).
    + rewrite Pos.eqb_sym. exact Heq.
    + apply andb_true_iff in Heq as [Hsym Hconst].
      rewrite Pos.eqb_sym, Hsym. apply static_const_eqb_flip, Hconst.
  - destruct c as [params my_closure ret exn metadata body].
    intros [params' my_closure' ret' exn' metadata' body']. simpl.
    pose proof (bind_names_flip params params' [(my_closure, my_closure')])
      as Hflip.
    change (flip [(my_closure, my_closure')])
      with [(my_closure', my_closure)] in Hflip.
    rewrite Hflip.
    destruct (bind_names params params' [(my_closure, my_closure')])
      as [vm | ]; [ | discriminate ].
    simpl. intros Heq. apply andb_true_iff in Heq as [Hmetadata Heq].
    apply code_metadata_eqb_eq in Hmetadata. subst metadata'.
    rewrite code_metadata_eqb_refl.
    exact (expr_eqb_flip body vm [(ret, ret'); (exn, exn')] _ Heq).
Qed.

(* ======================================================================== *)
(* Transitivity                                                             *)
(* ======================================================================== *)

(* What the first map matches and then the second, the third matches. *)
Definition composes (m1 m2 m : name_map) : Prop :=
  ∀ a a' a'' : positive,
    names_match m1 a a' = true -> names_match m2 a' a'' = true ->
    names_match m a a'' = true.

Lemma composes_nil : composes [] [] [].
Proof.
  intros a a' a'' H1 H2. apply names_match_nil in H1, H2. subst a' a''.
  apply Pos.eqb_refl.
Qed.

(* A map composes with the empty one. *)
Lemma composes_left (m : name_map) : composes m [] m.
Proof.
  intros a a' a'' H1 H2. apply names_match_nil in H2. subst a''. exact H1.
Qed.

Lemma composes_cons (m1 m2 m : name_map) (b b' b'' : positive) :
  composes m1 m2 m ->
  composes ((b, b') :: m1) ((b', b'') :: m2) ((b, b'') :: m).
Proof.
  intros Hm a a' a''. simpl. destruct (Pos.eqb_spec a b) as [-> | Hne].
  - intros Heq. apply Pos.eqb_eq in Heq. subst a'. rewrite Pos.eqb_refl.
    auto.
  - destruct (Pos.eqb_spec a' b') as [-> | Hne']; [ discriminate | ].
    intros H1. destruct (Pos.eqb a'' b''); [ discriminate | ].
    exact (Hm a a' a'' H1).
Qed.

Lemma bind_names_composes (xs : list positive) :
  ∀ (xs' xs'' : list positive) (m1 m2 m m1' m2' : name_map),
    composes m1 m2 m ->
    bind_names xs xs' m1 = Some m1' ->
    bind_names xs' xs'' m2 = Some m2' ->
    ∃ m' : name_map, bind_names xs xs'' m = Some m' ∧ composes m1' m2' m'.
Proof.
  induction xs as [ | x xs IH];
    intros [ | x' xs'] [ | x'' xs''] m1 m2 m m1' m2' Hm H1 H2; simpl in *;
    try discriminate H1; try discriminate H2.
  - injection H1 as <-. injection H2 as <-.
    exists m. split; [ reflexivity | exact Hm ].
  - exact (IH xs' xs'' _ _ _ m1' m2' (composes_cons _ _ _ x x' x'' Hm) H1
             H2).
Qed.

Lemma combine_composes (xs : list positive) :
  ∀ (xs' xs'' : list positive) (m1 m2 m : name_map),
    length xs = length xs' ->
    length xs' = length xs'' ->
    composes m1 m2 m ->
    composes (combine xs xs' ++ m1) (combine xs' xs'' ++ m2)
      (combine xs xs'' ++ m).
Proof.
  induction xs as [ | x xs IH];
    intros [ | x' xs'] [ | x'' xs''] m1 m2 m H1 H2 Hm; simpl in *;
    try discriminate H1; try discriminate H2.
  - exact Hm.
  - injection H1 as H1. injection H2 as H2.
    apply composes_cons, IH; assumption.
Qed.

Lemma combine_names_composes (xs xs' xs'' : list positive)
    (m1 m2 m m1' m2' : name_map) :
  composes m1 m2 m ->
  combine_names xs xs' m1 = Some m1' ->
  combine_names xs' xs'' m2 = Some m2' ->
  ∃ m' : name_map, combine_names xs xs'' m = Some m' ∧ composes m1' m2' m'.
Proof.
  unfold combine_names. intros Hm.
  destruct (Nat.eqb_spec (length xs) (length xs')) as [H1 | _];
    [ | discriminate ].
  destruct (Nat.eqb_spec (length xs') (length xs'')) as [H2 | _];
    [ | intros _; discriminate ].
  intros [= <-] [= <-].
  replace (Nat.eqb (length xs) (length xs'')) with true
    by (symmetry; apply Nat.eqb_eq; congruence).
  eexists. split; [ reflexivity | apply combine_composes; assumption ].
Qed.

Lemma forall2b_trans {A B C : Type} (f : A -> B -> bool)
    (g : B -> C -> bool) (h : A -> C -> bool) (l : list A) :
  (∀ (a : A) (b : B) (c : C),
    f a b = true -> g b c = true -> h a c = true) ->
  ∀ (l' : list B) (l'' : list C),
    forall2b f l l' = true -> forall2b g l' l'' = true ->
    forall2b h l l'' = true.
Proof.
  intros Hfgh. induction l as [ | a l IH];
    intros [ | b l'] [ | c l''] H1 H2; simpl in *;
    try discriminate H1; try discriminate H2; [ reflexivity | ].
  apply andb_true_iff in H1 as [Hab H1].
  apply andb_true_iff in H2 as [Hbc H2].
  rewrite (Hfgh a b c Hab Hbc), (IH l' l'' H1 H2). reflexivity.
Qed.

Section Composed.

Variables vm1 vm2 vm : name_map.

Hypothesis Hvm : composes vm1 vm2 vm.

Lemma simple_eqb_trans (s s' s'' : simple) :
  simple_eqb vm1 s s' = true -> simple_eqb vm2 s' s'' = true ->
  simple_eqb vm s s'' = true.
Proof.
  destruct s as [x | sym | c], s' as [x' | sym' | c'],
    s'' as [x'' | sym'' | c''];
    simpl; intros H1 H2; try discriminate H1; try discriminate H2.
  - exact (Hvm _ _ _ H1 H2).
  - apply Pos.eqb_eq in H1, H2. subst sym' sym''. apply Pos.eqb_refl.
  - apply const_eqb_eq in H1, H2. subst c' c''. apply const_eqb_refl.
Qed.

Lemma simples_eqb_trans (args args' args'' : list simple) :
  simples_eqb vm1 args args' = true -> simples_eqb vm2 args' args'' = true ->
  simples_eqb vm args args'' = true.
Proof. apply forall2b_trans. exact simple_eqb_trans. Qed.

Lemma value_slots_eqb_trans
    (slots slots' slots'' : list (value_slot * simple)) :
  value_slots_eqb vm1 slots slots' = true ->
  value_slots_eqb vm2 slots' slots'' = true ->
  value_slots_eqb vm slots slots'' = true.
Proof.
  apply forall2b_trans. intros slot slot' slot'' H1 H2.
  apply andb_true_iff in H1 as [Hslot1 Hs1].
  apply andb_true_iff in H2 as [Hslot2 Hs2].
  apply Pos.eqb_eq in Hslot1, Hslot2. rewrite Hslot1, Hslot2, Pos.eqb_refl.
  exact (simple_eqb_trans _ _ _ Hs1 Hs2).
Qed.

Lemma named_eqb_trans (d d' d'' : named) :
  named_eqb vm1 d d' = true -> named_eqb vm2 d' d'' = true ->
  named_eqb vm d d'' = true.
Proof.
  destruct d
    as [s | p args | code slots | fslot decls slots | slot closure
       | pfrom slot closure | move_from move_to closure],
    d'
    as [s' | p' args' | code' slots' | fslot' decls' slots' | slot' closure'
       | pfrom' slot' closure' | move_from' move_to' closure'];
    simpl; try discriminate; intros H1;
    destruct d''
      as [s'' | p'' args'' | code'' slots'' | fslot'' decls'' slots''
         | slot'' closure'' | pfrom'' slot'' closure''
         | move_from'' move_to'' closure''];
    simpl; try discriminate; intros H2.
  - exact (simple_eqb_trans _ _ _ H1 H2).
  - apply andb_true_iff in H1 as [Hp1 Hargs1].
    apply andb_true_iff in H2 as [Hp2 Hargs2].
    apply prim_eqb_sound in Hp1, Hp2. subst p' p''. rewrite prim_eqb_refl.
    exact (simples_eqb_trans _ _ _ Hargs1 Hargs2).
  - apply andb_true_iff in H1 as [Hcode1 Hslots1].
    apply andb_true_iff in H2 as [Hcode2 Hslots2].
    apply Pos.eqb_eq in Hcode1, Hcode2. subst code' code''.
    rewrite Pos.eqb_refl. exact (value_slots_eqb_trans _ _ _ Hslots1 Hslots2).
  - apply andb_true_iff in H1 as [H1 Hslots1].
    apply andb_true_iff in H1 as [Hfslot1 Hdecls1].
    apply andb_true_iff in H2 as [H2 Hslots2].
    apply andb_true_iff in H2 as [Hfslot2 Hdecls2].
    apply Pos.eqb_eq in Hfslot1, Hfslot2.
    apply names_eqb_eq in Hdecls1, Hdecls2.
    subst fslot' fslot'' decls' decls''.
    rewrite Pos.eqb_refl, names_eqb_refl.
    exact (value_slots_eqb_trans _ _ _ Hslots1 Hslots2).
  - apply andb_true_iff in H1 as [Hslot1 Hclosure1].
    apply andb_true_iff in H2 as [Hslot2 Hclosure2].
    apply Pos.eqb_eq in Hslot1, Hslot2. subst slot' slot''.
    rewrite Pos.eqb_refl. exact (simple_eqb_trans _ _ _ Hclosure1 Hclosure2).
  - apply andb_true_iff in H1 as [H1 Hclosure1].
    apply andb_true_iff in H1 as [Hfrom1 Hslot1].
    apply andb_true_iff in H2 as [H2 Hclosure2].
    apply andb_true_iff in H2 as [Hfrom2 Hslot2].
    apply Pos.eqb_eq in Hfrom1, Hfrom2, Hslot1, Hslot2.
    subst pfrom' pfrom'' slot' slot''. rewrite !Pos.eqb_refl.
    exact (simple_eqb_trans _ _ _ Hclosure1 Hclosure2).
  - apply andb_true_iff in H1 as [H1 Hclosure1].
    apply andb_true_iff in H1 as [Hfrom1 Hto1].
    apply andb_true_iff in H2 as [H2 Hclosure2].
    apply andb_true_iff in H2 as [Hfrom2 Hto2].
    apply Pos.eqb_eq in Hfrom1, Hfrom2, Hto1, Hto2.
    subst move_from' move_from'' move_to' move_to''. rewrite !Pos.eqb_refl.
    exact (simple_eqb_trans _ _ _ Hclosure1 Hclosure2).
Qed.

Lemma static_const_eqb_trans (const const' const'' : static_const) :
  static_const_eqb vm1 const const' = true ->
  static_const_eqb vm2 const' const'' = true ->
  static_const_eqb vm const const'' = true.
Proof.
  destruct const
    as [code slots | fslot syms decls slots | tag fields | s | bytes
        | elements],
    const'
    as [code' slots' | fslot' syms' decls' slots' | tag' fields' | s'
       | bytes' | elements'];
    simpl; try discriminate; intros H1;
    destruct const''
      as [code'' slots'' | fslot'' syms'' decls'' slots'' | tag'' fields''
         | s'' | bytes'' | elements''];
    simpl; try discriminate; intros H2.
  - apply andb_true_iff in H1 as [Hcode1 Hslots1].
    apply andb_true_iff in H2 as [Hcode2 Hslots2].
    apply Pos.eqb_eq in Hcode1, Hcode2. subst code' code''.
    rewrite Pos.eqb_refl. exact (value_slots_eqb_trans _ _ _ Hslots1 Hslots2).
  - apply andb_true_iff in H1 as [H1 Hslots1].
    apply andb_true_iff in H1 as [H1 Hdecls1].
    apply andb_true_iff in H1 as [Hfslot1 Hsyms1].
    apply andb_true_iff in H2 as [H2 Hslots2].
    apply andb_true_iff in H2 as [H2 Hdecls2].
    apply andb_true_iff in H2 as [Hfslot2 Hsyms2].
    apply Pos.eqb_eq in Hfslot1, Hfslot2.
    apply names_eqb_eq in Hsyms1, Hsyms2, Hdecls1, Hdecls2.
    subst fslot' fslot'' syms' syms'' decls' decls''.
    rewrite Pos.eqb_refl, !names_eqb_refl.
    exact (value_slots_eqb_trans _ _ _ Hslots1 Hslots2).
  - apply andb_true_iff in H1 as [Htag1 Hfields1].
    apply andb_true_iff in H2 as [Htag2 Hfields2].
    apply Z.eqb_eq in Htag1, Htag2. subst tag' tag''. rewrite Z.eqb_refl.
    exact (simples_eqb_trans _ _ _ Hfields1 Hfields2).
  - exact (simple_eqb_trans _ _ _ H1 H2).
  - apply forall2b_Z_eqb in H1, H2. subst bytes' bytes''.
    apply forall2b_refl, Z.eqb_refl.
  - exact (simples_eqb_trans _ _ _ H1 H2).
Qed.

Variables km1 km2 km : name_map.

Hypothesis Hkm : composes km1 km2 km.

Lemma apply_cont_eqb_trans (ac ac' ac'' : apply_cont) :
  apply_cont_eqb vm1 km1 ac ac' = true ->
  apply_cont_eqb vm2 km2 ac' ac'' = true ->
  apply_cont_eqb vm km ac ac'' = true.
Proof.
  unfold apply_cont_eqb. intros H1 H2.
  apply andb_true_iff in H1 as [H1 Htrap1].
  apply andb_true_iff in H1 as [Hk1 Hargs1].
  apply andb_true_iff in H2 as [H2 Htrap2].
  apply andb_true_iff in H2 as [Hk2 Hargs2].
  rewrite (Hkm _ _ _ Hk1 Hk2), (simples_eqb_trans _ _ _ Hargs1 Hargs2).
  simpl.
  destruct (ac_trap_action ac) as [[k | k kind] | ],
    (ac_trap_action ac') as [[k' | k' kind'] | ];
    simpl in Htrap1; try discriminate;
    destruct (ac_trap_action ac'') as [[k'' | k'' kind''] | ];
    simpl in Htrap2 |- *; try discriminate; try reflexivity.
  - exact (Hkm _ _ _ Htrap1 Htrap2).
  - apply andb_true_iff in Htrap1 as [Hk1' Hkind1].
    apply andb_true_iff in Htrap2 as [Hk2' Hkind2].
    rewrite (Hkm _ _ _ Hk1' Hk2'). simpl.
    destruct kind as [[ | | ] | ], kind' as [[ | | ] | ];
      simpl in Hkind1; try discriminate; exact Hkind2.
Qed.

Lemma apply_eqb_trans (ap ap' ap'' : apply_expr) :
  apply_eqb vm1 km1 ap ap' = true ->
  apply_eqb vm2 km2 ap' ap'' = true ->
  apply_eqb vm km ap ap'' = true.
Proof.
  unfold apply_eqb. intros H1 H2.
  apply andb_true_iff in H1 as [H1 Hreturn_arity1].
  apply andb_true_iff in H1 as [H1 Hargs_arity1].
  apply andb_true_iff in H1 as [H1 Hkinds1].
  apply andb_true_iff in H1 as [H1 Hextra1].
  apply andb_true_iff in H1 as [H1 Hexn1].
  apply andb_true_iff in H1 as [H1 Hreturn1].
  apply andb_true_iff in H1 as [H1 Hargs1].
  apply andb_true_iff in H1 as [Hcallee1 Hkind1].
  apply andb_true_iff in H2 as [H2 Hreturn_arity2].
  apply andb_true_iff in H2 as [H2 Hargs_arity2].
  apply andb_true_iff in H2 as [H2 Hkinds2].
  apply andb_true_iff in H2 as [H2 Hextra2].
  apply andb_true_iff in H2 as [H2 Hexn2].
  apply andb_true_iff in H2 as [H2 Hreturn2].
  apply andb_true_iff in H2 as [H2 Hargs2].
  apply andb_true_iff in H2 as [Hcallee2 Hkind2].
  apply call_kind_eqb_eq in Hkind1, Hkind2.
  apply kinds_eqb_eq in Hkinds1, Hkinds2.
  apply arity_eqb_eq in Hargs_arity1, Hargs_arity2, Hreturn_arity1,
    Hreturn_arity2.
  rewrite (Hkm _ _ _ Hexn1 Hexn2), (Hkm _ _ _ Hreturn1 Hreturn2),
    (simple_eqb_trans _ _ _ Hcallee1 Hcallee2),
    (simples_eqb_trans _ _ _ Hargs1 Hargs2),
    (simples_eqb_trans _ _ _ Hextra1 Hextra2), Hkind1, Hkind2,
    call_kind_eqb_refl, Hkinds1, Hkinds2, kinds_eqb_refl, Hargs_arity1,
    Hargs_arity2, Hreturn_arity1, Hreturn_arity2, !arity_eqb_refl.
  reflexivity.
Qed.

End Composed.

Fixpoint expr_eqb_trans (e : expr) :
  ∀ vm1 vm2 vm km1 km2 km : name_map,
    composes vm1 vm2 vm ->
    composes km1 km2 km ->
    ∀ e' : expr,
      expr_eqb vm1 km1 e e' = true ->
      ∀ e'' : expr,
        expr_eqb vm2 km2 e' e'' = true -> expr_eqb vm km e e'' = true
with handler_eqb_trans (h : continuation_handler) :
  ∀ vm1 vm2 vm km1 km2 km : name_map,
    composes vm1 vm2 vm ->
    composes km1 km2 km ->
    ∀ h' : continuation_handler,
      handler_eqb vm1 km1 h h' = true ->
      ∀ h'' : continuation_handler,
        handler_eqb vm2 km2 h' h'' = true -> handler_eqb vm km h h'' = true
with binding_eqb_trans (binding : static_binding) :
  ∀ vm1 vm2 vm : name_map,
    composes vm1 vm2 vm ->
    ∀ binding' : static_binding,
      binding_eqb vm1 binding binding' = true ->
      ∀ binding'' : static_binding,
        binding_eqb vm2 binding' binding'' = true ->
        binding_eqb vm binding binding'' = true
with code_eqb_trans (c : code) :
  ∀ c' : code,
    code_eqb c c' = true ->
    ∀ c'' : code, code_eqb c' c'' = true -> code_eqb c c'' = true.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros vm1 vm2 vm km1 km2 km Hvm Hkm
        [ x' kind' d' body' | bindings' body' | k' h' body' | handlers' body'
        | ap' | ac' | scrutinee' arms' | ];
      simpl; try discriminate; intros H1
        [ x'' kind'' d'' body'' | bindings'' body'' | k'' h'' body''
        | handlers'' body'' | ap'' | ac'' | scrutinee'' arms'' | ];
      simpl; try discriminate; intros H2.
    + (* Let *)
      apply andb_true_iff in H1 as [Hd1 Hbody1].
      apply andb_true_iff in Hd1 as [Hkind1 Hd1].
      apply andb_true_iff in H2 as [Hd2 Hbody2].
      apply andb_true_iff in Hd2 as [Hkind2 Hd2].
      apply kind_eqb_eq in Hkind1, Hkind2. subst kind' kind''.
      rewrite kind_eqb_refl, (named_eqb_trans _ _ _ Hvm _ _ _ Hd1 Hd2).
      exact (expr_eqb_trans body _ _ _ _ _ _
               (composes_cons _ _ _ x x' x'' Hvm) Hkm _ Hbody1 _ Hbody2).
    + (* Let_static *)
      apply andb_true_iff in H1 as [Hbindings1 Hbody1].
      apply andb_true_iff in H2 as [Hbindings2 Hbody2].
      rewrite (expr_eqb_trans body _ _ _ _ _ _ Hvm Hkm _ Hbody1 _ Hbody2),
        andb_true_r.
      clear Hbody1 Hbody2. revert bindings' Hbindings1 bindings'' Hbindings2.
      induction bindings as [ | binding bindings IH];
        intros [ | binding' bindings'] H1 [ | binding'' bindings''] H2;
        simpl in *; try discriminate H1; try discriminate H2;
        [ reflexivity | ].
      apply andb_true_iff in H1 as [Hfirst1 Hrest1].
      apply andb_true_iff in H2 as [Hfirst2 Hrest2].
      rewrite (binding_eqb_trans binding _ _ _ Hvm _ Hfirst1 _ Hfirst2),
        (IH _ Hrest1 _ Hrest2).
      reflexivity.
    + (* Let_cont_non_recursive *)
      apply andb_true_iff in H1 as [Hh1 Hbody1].
      apply andb_true_iff in H2 as [Hh2 Hbody2].
      rewrite (handler_eqb_trans h _ _ _ _ _ _ Hvm Hkm _ Hh1 _ Hh2).
      exact (expr_eqb_trans body _ _ _ _ _ _ Hvm
               (composes_cons _ _ _ k k' k'' Hkm) _ Hbody1 _ Hbody2).
    + (* Let_cont_recursive *)
      revert H1 H2.
      destruct (combine_names (map fst handlers) (map fst handlers') km1)
        as [km1' | ] eqn:Hnames1; [ | discriminate ].
      destruct (combine_names (map fst handlers') (map fst handlers'') km2)
        as [km2' | ] eqn:Hnames2; [ | intros _; discriminate ].
      intros H1 H2.
      destruct (combine_names_composes _ _ _ _ _ _ _ _ Hkm Hnames1 Hnames2)
        as (km' & -> & Hkm').
      apply andb_true_iff in H1 as [Hhandlers1 Hbody1].
      apply andb_true_iff in H2 as [Hhandlers2 Hbody2].
      rewrite (expr_eqb_trans body _ _ _ _ _ _ Hvm Hkm' _ Hbody1 _ Hbody2),
        andb_true_r.
      clear Hbody1 Hbody2 Hnames1 Hnames2.
      revert handlers' Hhandlers1 handlers'' Hhandlers2.
      induction handlers as [ | [k h] handlers IH];
        intros [ | [k' h'] handlers'] H1 [ | [k'' h''] handlers''] H2;
        simpl in *; try discriminate H1; try discriminate H2;
        [ reflexivity | ].
      apply andb_true_iff in H1 as [Hfirst1 Hrest1].
      apply andb_true_iff in H2 as [Hfirst2 Hrest2].
      rewrite (handler_eqb_trans h _ _ _ _ _ _ Hvm Hkm' _ Hfirst1 _ Hfirst2),
        (IH _ Hrest1 _ Hrest2).
      reflexivity.
    + (* Apply *)
      exact (apply_eqb_trans _ _ _ Hvm _ _ _ Hkm _ _ _ H1 H2).
    + (* Apply_cont *)
      exact (apply_cont_eqb_trans _ _ _ Hvm _ _ _ Hkm _ _ _ H1 H2).
    + (* Switch *)
      apply andb_true_iff in H1 as [Hscrutinee1 Harms1].
      apply andb_true_iff in H2 as [Hscrutinee2 Harms2].
      rewrite (simple_eqb_trans _ _ _ Hvm _ _ _ Hscrutinee1 Hscrutinee2).
      refine (forall2b_trans _ _ _ _ _ _ _ Harms1 Harms2).
      intros arm arm' arm'' Harm1 Harm2.
      apply andb_true_iff in Harm1 as [Hz1 Hac1].
      apply andb_true_iff in Harm2 as [Hz2 Hac2].
      apply Z.eqb_eq in Hz1, Hz2. rewrite Hz1, Hz2, Z.eqb_refl.
      exact (apply_cont_eqb_trans _ _ _ Hvm _ _ _ Hkm _ _ _ Hac1 Hac2).
    + (* Invalid *)
      reflexivity.
  - destruct h as [params metadata handler].
    intros vm1 vm2 vm km1 km2 km Hvm Hkm [params' metadata' handler'] H1
      [params'' metadata'' handler''] H2.
    simpl in *. revert H1 H2.
    destruct (bind_names params params' vm1) as [vm1' | ] eqn:Hbind1;
      [ | discriminate ].
    destruct (bind_names params' params'' vm2) as [vm2' | ] eqn:Hbind2;
      [ | intros _; discriminate ].
    intros H1 H2.
    destruct (bind_names_composes params _ _ _ _ _ _ _ Hvm Hbind1 Hbind2)
      as (vm' & -> & Hvm').
    apply andb_true_iff in H1 as [Hmetadata1 H1].
    apply andb_true_iff in H2 as [Hmetadata2 H2].
    apply handler_metadata_eqb_eq in Hmetadata1, Hmetadata2.
    subst metadata' metadata''. rewrite handler_metadata_eqb_refl.
    exact (expr_eqb_trans handler _ _ _ _ _ _ Hvm' Hkm _ H1 _ H2).
  - destruct binding as [id c | id | sym const];
      intros vm1 vm2 vm Hvm [id' c' | id' | sym' const']; simpl;
      try discriminate; intros H1 [id'' c'' | id'' | sym'' const'']; simpl;
      try discriminate; intros H2.
    + apply andb_true_iff in H1 as [Hid1 Hc1].
      apply andb_true_iff in H2 as [Hid2 Hc2].
      apply Pos.eqb_eq in Hid1, Hid2. subst id' id''. rewrite Pos.eqb_refl.
      exact (code_eqb_trans c _ Hc1 _ Hc2).
    + apply Pos.eqb_eq in H1, H2. subst id' id''. apply Pos.eqb_refl.
    + apply andb_true_iff in H1 as [Hsym1 Hconst1].
      apply andb_true_iff in H2 as [Hsym2 Hconst2].
      apply Pos.eqb_eq in Hsym1, Hsym2. subst sym' sym''.
      rewrite Pos.eqb_refl.
      exact (static_const_eqb_trans _ _ _ Hvm _ _ _ Hconst1 Hconst2).
  - destruct c as [params my_closure ret exn metadata body].
    intros [params' my_closure' ret' exn' metadata' body'] H1
      [params'' my_closure'' ret'' exn'' metadata'' body''] H2.
    simpl in *. revert H1 H2.
    destruct (bind_names params params' [(my_closure, my_closure')])
      as [vm1 | ] eqn:Hbind1; [ | discriminate ].
    destruct (bind_names params' params'' [(my_closure', my_closure'')])
      as [vm2 | ] eqn:Hbind2; [ | intros _; discriminate ].
    intros H1 H2.
    destruct (bind_names_composes params _ _ _ _
                [(my_closure, my_closure'')] _ _
                (composes_cons _ _ _ my_closure my_closure' my_closure''
                   composes_nil)
                Hbind1 Hbind2)
      as (vm & -> & Hvm).
    apply andb_true_iff in H1 as [Hmetadata1 H1].
    apply andb_true_iff in H2 as [Hmetadata2 H2].
    apply code_metadata_eqb_eq in Hmetadata1, Hmetadata2.
    subst metadata' metadata''. rewrite code_metadata_eqb_refl.
    exact (expr_eqb_trans body _ _ _ _ _ _ Hvm
             (composes_cons _ _ _ ret ret' ret''
                (composes_cons _ _ _ exn exn' exn'' composes_nil))
             _ H1 _ H2).
Qed.

(* ======================================================================== *)
(* Contexts                                                                 *)
(* ======================================================================== *)

(* Under a context the names bound around the hole are paired with
   themselves; the comparison with no names paired composes with that of
   an expression with itself under those pairs. *)
Lemma plug_eqb (c : context) :
  ∀ (vm km : name_map) (e e' : expr),
    reflexive_map vm ->
    reflexive_map km ->
    expr_eqb [] [] e e' = true ->
    expr_eqb vm km (plug c e) (plug c e') = true.
Proof.
  induction c
    as [ | x kind d c IH | bindings c IH
       | before id params my_closure ret exn metadata c IH after body
       | k h c IH | k params metadata c IH body | handlers c IH
       | before k params metadata c IH after body ];
    intros vm km e e' Hvm Hkm Heq; simpl.
  - exact (expr_eqb_trans e vm [] vm km [] km (composes_left vm)
             (composes_left km) e (expr_eqb_refl e vm km Hvm Hkm) e' Heq).
  - rewrite kind_eqb_refl, named_eqb_refl by exact Hvm.
    apply IH; [ apply reflexive_map_cons, Hvm | exact Hkm | exact Heq ].
  - rewrite (IH vm km e e' Hvm Hkm Heq), andb_true_r.
    apply forall2b_refl. intros binding. apply binding_eqb_refl, Hvm.
  - rewrite (expr_eqb_refl body vm km Hvm Hkm), andb_true_r.
    assert (Hsame :
      ∀ bindings : list static_binding,
        forall2b (binding_eqb vm) bindings bindings = true)
      by (intros bindings; apply forall2b_refl; intros binding;
          apply binding_eqb_refl, Hvm).
    apply forall2b_app; [ apply Hsame | ]. simpl.
    rewrite Pos.eqb_refl, Hsame, andb_true_r. simpl.
    destruct (bind_names_refl params [(my_closure, my_closure)]
                (reflexive_map_cons _ _ reflexive_map_nil))
      as (vm0 & -> & Hvm0).
    rewrite code_metadata_eqb_refl.
    apply IH; [ exact Hvm0 | | exact Heq ].
    apply reflexive_map_cons, reflexive_map_cons, reflexive_map_nil.
  - rewrite handler_eqb_refl by assumption.
    apply IH; [ exact Hvm | apply reflexive_map_cons, Hkm | exact Heq ].
  - rewrite (expr_eqb_refl body vm ((k, k) :: km) Hvm
               (reflexive_map_cons _ _ Hkm)),
      andb_true_r.
    destruct (bind_names_refl params vm Hvm) as (vm0 & -> & Hvm0).
    rewrite handler_metadata_eqb_refl. apply IH; assumption.
  - destruct (combine_names_refl (map fst handlers) km Hkm)
      as (km' & -> & Hkm').
    rewrite (IH vm km' e e' Hvm Hkm' Heq), andb_true_r.
    apply forall2b_refl. intros [k0 h0]. apply handler_eqb_refl; assumption.
  - rewrite !map_app. simpl.
    destruct (combine_names_refl (map fst before ++ k :: map fst after) km
                Hkm)
      as (km' & -> & Hkm').
    rewrite (expr_eqb_refl body vm km' Hvm Hkm'), andb_true_r.
    assert (Hsame :
      ∀ handlers : list (continuation * continuation_handler),
        forall2b (fun '(_, h) '(_, h') => handler_eqb vm km' h h') handlers
          handlers
        = true)
      by (intros handlers; apply forall2b_refl; intros [k0 h0];
          apply handler_eqb_refl; assumption).
    apply forall2b_app; [ apply Hsame | ]. simpl.
    rewrite Hsame, andb_true_r.
    destruct (bind_names_refl params vm Hvm) as (vm0 & -> & Hvm0).
    rewrite handler_metadata_eqb_refl. apply IH; assumption.
Qed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Theorem expr_eqb_reflexive : expr_eqb_reflexive_statement.
Proof. intros e. apply expr_eqb_refl; apply reflexive_map_nil. Qed.

Theorem expr_eqb_symmetric : expr_eqb_symmetric_statement.
Proof. intros e e' Heq. exact (expr_eqb_flip e [] [] e' Heq). Qed.

Theorem expr_eqb_transitive : expr_eqb_transitive_statement.
Proof.
  intros e e' e'' H1 H2.
  exact (expr_eqb_trans e [] [] [] [] [] [] composes_nil composes_nil e' H1
           e'' H2).
Qed.

Theorem alpha_equivalent_accepted : alpha_equivalent_accepted_statement.
Proof.
  intros e e' Halpha.
  induction Halpha
    as [ e | e e' _ IH | e e' e'' _ IH _ IH' | c e e' _ IH
       | r e Hclosed Hvars Hconts | r x kind d body Hclosed Hvars Hconts
       | r k h body Hclosed Hvars Hconts
       | r k params metadata handler body Hclosed Hvars Hconts
       | r before k params metadata handler after body Hclosed Hvars Hconts
       | r before id c after body Hclosed ]
    using alpha_equivalent_by_relation_ind.
  - apply expr_eqb_reflexive.
  - apply expr_eqb_symmetric, IH.
  - exact (expr_eqb_transitive _ _ _ IH IH').
  - apply plug_eqb; [ apply reflexive_map_nil | apply reflexive_map_nil | ].
    exact IH.
  - apply rename_expr_accepted; assumption.
  - apply rename_let_accepted; assumption.
  - apply rename_let_cont_accepted; assumption.
  - apply rename_handler_params_accepted; assumption.
  - apply rename_recursive_handler_params_accepted; assumption.
  - apply rename_code_accepted, Hclosed.
Qed.

(* A handler that repeats its parameter, and one that does not. The
   later of two parameters of one name is the one in scope. *)
Definition repeats_parameter : expr :=
  Let_cont_non_recursive 1%positive
    (Handler [2%positive; 2%positive] (make_handler_metadata [Value; Value])
       (Apply_cont (Apply_cont_expr 3%positive [Var 2%positive] None)))
    Invalid.

Definition repeats_no_parameter : expr :=
  Let_cont_non_recursive 1%positive
    (Handler [2%positive; 4%positive] (make_handler_metadata [Value; Value])
       (Apply_cont (Apply_cont_expr 3%positive [Var 4%positive] None)))
    Invalid.

Theorem expr_eqb_repeated_parameter : expr_eqb_repeated_parameter_statement.
Proof.
  exists repeats_parameter, repeats_no_parameter.
  split; [ reflexivity | ]. split; [ reflexivity | ]. split.
  - intros r Heq. unfold repeats_parameter, repeats_no_parameter in Heq.
    simpl in Heq. injection Heq as _ Hfirst Hsecond _. congruence.
  - intros Halpha. apply alpha_equivalent_binders_distinct in Halpha.
    discriminate Halpha.
Qed.

(* Code that mentions a variable it does not bind, under the binder of
   that variable; and the same with the binder and its use outside the
   code renamed. *)
Definition open_code : code :=
  Code [] 5%positive 6%positive 7%positive
    (make_code_metadata [] [Singleton Value])
    (Apply_cont (Apply_cont_expr 6%positive [Var 2%positive] None)).

Definition binds_around_open_code (x : variable) : expr :=
  Let x Value (Simple (Const (Tagged_immediate 0%Z)))
    (Let_static [Bind_code 1%positive open_code]
       (Apply_cont (Apply_cont_expr 3%positive [Var x] None))).

(* The variables of the [Let]s, down the bodies, whose body has code
   that is not closed. No rule of [alpha_equivalent] renames one, so
   alpha-equivalent terms have the same. *)
Fixpoint open_lets (e : expr) : list variable :=
  match e with
  | Let x _ _ body =>
    (if code_closed body then [] else [x]) ++ open_lets body
  | Let_static _ body | Let_cont_non_recursive _ _ body
  | Let_cont_recursive _ body =>
    open_lets body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => []
  end.

Lemma closed_open_lets (e : expr) :
  code_closed e = true -> open_lets e = [].
Proof.
  induction e
    as [ x kind d body IH | bindings body IH | k h body IH | handlers body IH
       | ap | ac | scrutinee arms | ];
    simpl; intros Hclosed; try reflexivity.
  - rewrite Hclosed. apply IH, Hclosed.
  - apply andb_true_iff in Hclosed. apply IH, Hclosed.
  - apply andb_true_iff in Hclosed. apply IH, Hclosed.
  - apply andb_true_iff in Hclosed. apply IH, Hclosed.
Qed.

Lemma plug_open_lets (c : context) (e e' : expr) :
  alpha_equivalent e e' ->
  open_lets e = open_lets e' ->
  open_lets (plug c e) = open_lets (plug c e').
Proof.
  intros Halpha Hsame.
  induction c
    as [ | y kind d c IH | bindings c IH
       | before id params my_closure ret exn metadata c IH after body
       | k h c IH
       | k params metadata c IH body | handlers c IH
       | before k params metadata c IH after body ];
    simpl; try reflexivity; try assumption.
  rewrite (alpha_equivalent_code_closed _ _ (Alpha_context c e e' Halpha)),
    IH.
  reflexivity.
Qed.

Lemma alpha_equivalent_open_lets (e e' : expr) :
  alpha_equivalent e e' -> open_lets e = open_lets e'.
Proof.
  intros Halpha.
  induction Halpha
    as [ e | e e' _ IH | e e' e'' _ IH _ IH' | c e e' Halpha IH
       | s e Hclosed _ _ | s x kind d body Hclosed _ _ | s k h body Hclosed _ _
       | s k params metadata handler body _ _ _
       | s before k params metadata handler after body _ _ _
       | s before id c after body _ ]
    using alpha_equivalent_by_relation_ind;
    try reflexivity.
  - symmetry. exact IH.
  - rewrite IH. exact IH'.
  - apply plug_open_lets; assumption.
  - rewrite !closed_open_lets;
      [ reflexivity | rewrite code_closed_renamed | ]; exact Hclosed.
  - simpl. rewrite code_closed_renamed, Hclosed. simpl.
    rewrite !closed_open_lets;
      [ reflexivity | rewrite code_closed_renamed | ]; exact Hclosed.
  - simpl.
    rewrite !closed_open_lets;
      [ reflexivity | rewrite code_closed_renamed | ]; exact Hclosed.
Qed.

Theorem expr_eqb_open_code : expr_eqb_open_code_statement.
Proof.
  exists (binds_around_open_code 2%positive),
    (binds_around_open_code 4%positive).
  repeat (split; [ reflexivity | ]). split.
  - intros r Heq. unfold binds_around_open_code, open_code in Heq.
    simpl in Heq. injection Heq as Hbinder Hcode _. congruence.
  - intros Halpha. apply alpha_equivalent_open_lets in Halpha.
    discriminate Halpha.
Qed.

(* ======================================================================== *)
(* The comparison does not depend on the names                              *)
(* ======================================================================== *)

Lemma forall2b_map_both {A B A' B' : Type} (g : A' -> B' -> bool)
    (h : A -> A') (h' : B -> B') (l : list A) :
  ∀ l' : list B,
    forall2b g (map h l) (map h' l')
    = forall2b (fun a b => g (h a) (h' b)) l l'.
Proof.
  induction l as [ | a l IH]; intros [ | b l']; simpl; try reflexivity.
  rewrite IH. reflexivity.
Qed.

Lemma forall2b_ext {A B : Type} (g g' : A -> B -> bool) (l : list A) :
  (∀ (a : A) (b : B), g a b = g' a b) ->
  ∀ l' : list B, forall2b g l l' = forall2b g' l l'.
Proof.
  intros Hsame. induction l as [ | a l IH]; intros [ | b l']; simpl;
    try reflexivity.
  rewrite Hsame, IH. reflexivity.
Qed.

Section Names_renamed.

Variable f : positive -> positive.

Hypothesis f_eqb : ∀ x y : positive, Pos.eqb (f x) (f y) = Pos.eqb x y.

Lemma names_match_renamed (m : name_map) (a a' : positive) :
  names_match (rename_name_map f m) (f a) (f a') = names_match m a a'.
Proof.
  induction m as [ | [b b'] m IH]; simpl; [ apply f_eqb | ].
  rewrite !f_eqb, IH. reflexivity.
Qed.

Lemma bind_names_renamed (xs : list positive) :
  ∀ (xs' : list positive) (m : name_map),
    bind_names (map f xs) (map f xs') (rename_name_map f m)
    = option_map (rename_name_map f) (bind_names xs xs' m).
Proof.
  induction xs as [ | x xs IH]; intros [ | x' xs'] m; simpl;
    try reflexivity.
  exact (IH xs' ((x, x') :: m)).
Qed.

Lemma combine_renamed (xs : list positive) :
  ∀ xs' : list positive,
    combine (map f xs) (map f xs') = rename_name_map f (combine xs xs').
Proof.
  induction xs as [ | x xs IH]; intros [ | x' xs']; simpl; try reflexivity.
  rewrite IH. reflexivity.
Qed.

Lemma combine_names_renamed (xs xs' : list positive) (m : name_map) :
  combine_names (map f xs) (map f xs') (rename_name_map f m)
  = option_map (rename_name_map f) (combine_names xs xs' m).
Proof.
  unfold combine_names. rewrite !length_map.
  destruct (Nat.eqb (length xs) (length xs')); [ | reflexivity ].
  simpl. rewrite combine_renamed. unfold rename_name_map.
  rewrite map_app. reflexivity.
Qed.

End Names_renamed.

Section Comparison_renamed.

Variable r : renaming.

Let vf := rename_variable r.

Let kf := rename_continuation r.

Lemma simple_eqb_equivariant (vm : name_map) (s s' : simple) :
  simple_eqb (rename_name_map vf vm) (rename_simple r s) (rename_simple r s')
  = simple_eqb vm s s'.
Proof.
  destruct s as [x | sym | c]; destruct s' as [x' | sym' | c']; simpl;
    try reflexivity.
  apply names_match_renamed, rename_variable_eqb.
Qed.

Lemma simples_eqb_equivariant (vm : name_map) (args args' : list simple) :
  simples_eqb (rename_name_map vf vm) (map (rename_simple r) args)
    (map (rename_simple r) args')
  = simples_eqb vm args args'.
Proof.
  unfold simples_eqb. rewrite forall2b_map_both.
  apply forall2b_ext. intros s s'. apply simple_eqb_equivariant.
Qed.

Lemma value_slots_eqb_equivariant (vm : name_map)
    (slots slots' : list (value_slot * simple)) :
  value_slots_eqb (rename_name_map vf vm)
    (map (fun slot => (fst slot, rename_simple r (snd slot))) slots)
    (map (fun slot => (fst slot, rename_simple r (snd slot))) slots')
  = value_slots_eqb vm slots slots'.
Proof.
  unfold value_slots_eqb. rewrite forall2b_map_both.
  apply forall2b_ext. intros slot slot'. simpl.
  rewrite simple_eqb_equivariant. reflexivity.
Qed.

Lemma named_eqb_equivariant (vm : name_map) (d d' : named) :
  named_eqb (rename_name_map vf vm) (rename_named r d) (rename_named r d')
  = named_eqb vm d d'.
Proof.
  destruct d; destruct d'; simpl; try reflexivity;
    rewrite ?simple_eqb_equivariant, ?simples_eqb_equivariant,
      ?value_slots_eqb_equivariant;
    reflexivity.
Qed.

Lemma static_const_eqb_equivariant (vm : name_map)
    (const const' : static_const) :
  static_const_eqb (rename_name_map vf vm) (rename_static_const r const)
    (rename_static_const r const')
  = static_const_eqb vm const const'.
Proof.
  destruct const; destruct const'; simpl; try reflexivity;
    rewrite ?simple_eqb_equivariant, ?simples_eqb_equivariant,
      ?value_slots_eqb_equivariant;
    reflexivity.
Qed.

Lemma trap_action_eqb_equivariant (km : name_map)
    (action action' : option trap_action) :
  trap_action_eqb (rename_name_map kf km)
    (option_map (rename_trap_action r) action)
    (option_map (rename_trap_action r) action')
  = trap_action_eqb km action action'.
Proof.
  destruct action as [[k | k kind] | ], action' as [[k' | k' kind'] | ];
    simpl; try reflexivity;
    rewrite (names_match_renamed kf (rename_continuation_eqb r));
    reflexivity.
Qed.

Lemma apply_cont_eqb_equivariant (vm km : name_map) (ac ac' : apply_cont) :
  apply_cont_eqb (rename_name_map vf vm) (rename_name_map kf km)
    (rename_apply_cont r ac) (rename_apply_cont r ac')
  = apply_cont_eqb vm km ac ac'.
Proof.
  unfold apply_cont_eqb. simpl.
  rewrite simples_eqb_equivariant, trap_action_eqb_equivariant.
  rewrite (names_match_renamed kf (rename_continuation_eqb r)).
  reflexivity.
Qed.

Lemma apply_eqb_equivariant (vm km : name_map) (ap ap' : apply_expr) :
  apply_eqb (rename_name_map vf vm) (rename_name_map kf km)
    (rename_apply r ap) (rename_apply r ap')
  = apply_eqb vm km ap ap'.
Proof.
  unfold apply_eqb. rewrite !exn_extra_simples_rename. simpl.
  rewrite simple_eqb_equivariant, !simples_eqb_equivariant.
  rewrite !(names_match_renamed kf (rename_continuation_eqb r)).
  rewrite !map_map. reflexivity.
Qed.

Fixpoint expr_eqb_renamed (e : expr) {struct e} :
  ∀ (vm km : name_map) (e' : expr),
    expr_eqb (rename_name_map vf vm) (rename_name_map kf km)
      (rename_expr r e) (rename_expr r e')
    = expr_eqb vm km e e'
with handler_eqb_renamed (h : continuation_handler) {struct h} :
  ∀ (vm km : name_map) (h' : continuation_handler),
    handler_eqb (rename_name_map vf vm) (rename_name_map kf km)
      (rename_handler r h) (rename_handler r h')
    = handler_eqb vm km h h'
with binding_eqb_renamed (binding : static_binding) {struct binding} :
  ∀ (vm : name_map) (binding' : static_binding),
    binding_eqb (rename_name_map vf vm) (rename_static_binding r binding)
      (rename_static_binding r binding')
    = binding_eqb vm binding binding'
with code_eqb_renamed (c : code) {struct c} :
  ∀ c' : code, code_eqb (rename_code r c) (rename_code r c') = code_eqb c c'.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros vm km
        [ x' kind' d' body' | bindings' body' | k' h' body' | handlers' body'
        | ap' | ac' | scrutinee' arms' | ];
      simpl; try reflexivity.
    + (* Let *)
      rewrite named_eqb_equivariant. f_equal.
      exact (expr_eqb_renamed body ((x, x') :: vm) km body').
    + (* Let_static *)
      rewrite (expr_eqb_renamed body vm km body'). f_equal.
      revert bindings'.
      induction bindings as [ | binding bindings IH];
        intros [ | binding' bindings']; simpl; try reflexivity.
      rewrite (binding_eqb_renamed binding vm binding'), IH. reflexivity.
    + (* Let_cont_non_recursive *)
      rewrite (handler_eqb_renamed h vm km h'). f_equal.
      exact (expr_eqb_renamed body vm ((k, k') :: km) body').
    + (* Let_cont_recursive *)
      fold (rename_handlers r handlers). fold (rename_handlers r handlers').
      rewrite !map_fst_rename_handlers.
      rewrite (combine_names_renamed kf).
      destruct (combine_names (map fst handlers) (map fst handlers') km)
        as [km' | ]; simpl; [ | reflexivity ].
      rewrite (expr_eqb_renamed body vm km' body'). f_equal.
      unfold rename_handlers. revert handlers'.
      induction handlers as [ | [k0 h] handlers IH];
        intros [ | [k0' h'] handlers']; simpl; try reflexivity.
      rewrite (handler_eqb_renamed h vm km' h'), IH. reflexivity.
    + (* Apply *)
      apply apply_eqb_equivariant.
    + (* Apply_cont *)
      apply apply_cont_eqb_equivariant.
    + (* Switch *)
      rewrite simple_eqb_equivariant. f_equal.
      rewrite forall2b_map_both. apply forall2b_ext. intros arm arm'.
      simpl. rewrite apply_cont_eqb_equivariant. reflexivity.
  - destruct h as [params metadata handler].
    intros vm km [params' metadata' handler']. simpl.
    rewrite (bind_names_renamed vf).
    destruct (bind_names params params' vm) as [vm' | ]; simpl;
      [ | reflexivity ].
    f_equal. exact (expr_eqb_renamed handler vm' km handler').
  - destruct binding as [id c | id | sym const];
      intros vm [id' c' | id' | sym' const']; simpl; try reflexivity.
    + rewrite (code_eqb_renamed c c'). reflexivity.
    + rewrite static_const_eqb_equivariant. reflexivity.
  - destruct c as [params my_closure ret exn metadata body].
    intros [params' my_closure' ret' exn' metadata' body']. simpl.
    change [(rename_variable r my_closure, rename_variable r my_closure')]
      with (rename_name_map vf [(my_closure, my_closure')]).
    rewrite (bind_names_renamed vf).
    destruct (bind_names params params' [(my_closure, my_closure')])
      as [vm' | ]; simpl; [ | reflexivity ].
    f_equal.
    exact (expr_eqb_renamed body vm' [(ret, ret'); (exn, exn')] body').
Qed.

End Comparison_renamed.

Theorem expr_eqb_equivariant : expr_eqb_equivariant_statement.
Proof.
  intros r vm km e e'. apply expr_eqb_renamed.
Qed.
