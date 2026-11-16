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


(* Proofs of the statements of FlambdaTypingEnvExamplesSpec.v. Nothing
   here needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs
  FlambdaBindingTime FlambdaEnvAliases FlambdaEnvAliasesSpec
  FlambdaEnvAliasesProofs FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaTypingEnvProofs FlambdaTypingEnvExamplesSpec.
Import ListNotations.
Local Open Scope Z_scope.

(* ======================================================================== *)
(* What Simplify builds for a small term                                    *)
(* ======================================================================== *)

(* A closure of the code [f] whose value slot [w] holds [v] has the type
   of [c], where [c] is that closure and [p] is [v]. *)
Lemma example_closure_type_satisfied (vars : env) (v : value) :
  lookup example_c vars
  = Some (Value_closure example_f [(example_w, v)]) ->
  lookup example_p vars = Some v ->
  kind_of_value v = Value ->
  type_satisfied empty_static_env vars
    (Value_closure example_f [(example_w, v)]) example_closure_type.
Proof.
  intros Hc Hp Hkind. apply exactly_this_closure_satisfied.
  exists [(unnamed_function_slot, example_f)], [(example_w, v)].
  split; [ reflexivity | ].
  split; [ intros name; reflexivity | ].
  split; [ intros name; reflexivity | ].
  apply closures_entry_lookup_iff. split; [ | split ].
  - intros g function_type code Hfunction Hcode.
    unfold unnamed_function_slot in *. cbn -[Pos.eqb] in Hfunction, Hcode.
    destruct (Pos.eqb g 1); [ | discriminate ].
    injection Hfunction as <-. injection Hcode as <-. reflexivity.
  - intros g ty closure Hty Hproject.
    unfold unnamed_function_slot in *. cbn -[Pos.eqb] in Hty.
    destruct (Pos.eqb_spec g 1) as [-> | ]; [ | discriminate ].
    injection Hty as <-. simpl in Hproject. injection Hproject as <-.
    apply (proj2 (alias_satisfied _ _ _ Value (Var example_c))).
    split; [ exact Hc | reflexivity ].
  - intros w ty contents Hty Hcontents. unfold example_w in *.
    cbn -[Pos.eqb] in Hty, Hcontents.
    destruct (Pos.eqb w 1); [ | discriminate ].
    injection Hty as <-. injection Hcontents as <-.
    apply (proj2 (alias_satisfied _ _ _ Value (Var example_p))).
    split; [ exact Hp | exact Hkind ].
Qed.

(* An environment respects a structure given by its three maps if it
   respects each binding of each. *)
Lemma aliases_respected_of_bindings (t : Aliases.t) (vars : env) :
  (∀ (n : name) (c : simple),
    (n, c) ∈ Aliases.canonical_elements t ->
    value_of_name vars n = eval_simple vars c) ->
  (∀ (c : name) (aliases : Aliases.Aliases_of_canonical_element.t)
     (n : name) (mode : Name_mode.t),
    (c, aliases) ∈ Aliases.aliases_of_canonical_names t ->
    (n, mode) ∈ aliases ->
    value_of_name vars n = value_of_name vars c) ->
  (∀ (k : const) (aliases : Aliases.Aliases_of_canonical_element.t)
     (n : name) (mode : Name_mode.t),
    (k, aliases) ∈ Aliases.aliases_of_consts t ->
    (n, mode) ∈ aliases ->
    value_of_name vars n = Some (Value_const k)) ->
  aliases_respected t vars.
Proof.
  intros Hcanonical Hnames Hconsts. split.
  - intros n c Hdemoted. apply Hcanonical.
    exact (keyed_lookup_in name_eqb name_eqb_eq n c _ Hdemoted).
  - intros c n Hlisted. unfold listed_alias in Hlisted.
    apply mem_lookup in Hlisted. destruct Hlisted as [mode Hmode].
    apply (keyed_lookup_in name_eqb name_eqb_eq) in Hmode.
    unfold Aliases.get_aliases_of_canonical_element in Hmode.
    destruct c as [x | s | k].
    + destruct (Keyed.lookup name_eqb (Name_var x)
                  (Aliases.aliases_of_canonical_names t))
        as [aliases | ] eqn:Haliases; [ | destruct Hmode ].
      apply (keyed_lookup_in name_eqb name_eqb_eq) in Haliases.
      exact (Hnames (Name_var x) aliases n mode Haliases Hmode).
    + destruct (Keyed.lookup name_eqb (Name_symbol s)
                  (Aliases.aliases_of_canonical_names t))
        as [aliases | ] eqn:Haliases; [ | destruct Hmode ].
      apply (keyed_lookup_in name_eqb name_eqb_eq) in Haliases.
      exact (Hnames (Name_symbol s) aliases n mode Haliases Hmode).
    + destruct (Keyed.lookup const_eqb k (Aliases.aliases_of_consts t))
        as [aliases | ] eqn:Haliases; [ | destruct Hmode ].
      apply (keyed_lookup_in const_eqb const_eqb_eq) in Haliases.
      exact (Hconsts k aliases n mode Haliases Hmode).
Qed.

Lemma example_env_typing_env_satisfied (exempt : name -> Prop)
    (tenv : Typing_env.t) (vp : value) :
  example_env = Some tenv ->
  has_type_unless (exempt (Name_var example_p)) empty_static_env
    (example_vars vp vp) vp (these_tagged_immediates [7; 8]) ->
  typing_env_satisfied_but exempt empty_static_env (example_vars vp vp) tenv.
Proof.
  intros Henv Hp. vm_compute in Henv. injection Henv as <-.
  assert (Hkind : kind_of_value vp = Value)
    by exact (has_type_unless_kind _ _ _ _ _ Hp).
  split; [ reflexivity | ]. split.
  - intros n ty btm v Hlookup Hvalue.
    apply (keyed_lookup_in name_eqb name_eqb_eq) in Hlookup.
    simpl in Hlookup.
    repeat (destruct Hlookup as [Hlookup | Hlookup];
            [ injection Hlookup as <- <- <- | ]); [ .. | contradiction ].
    all: cbv in Hvalue; injection Hvalue as <-.
    all: try (right; apply (proj2 (unknown_satisfied _ _ _ Value));
              first [ exact Hkind | reflexivity ]).
    + right. apply (proj2 (alias_satisfied _ _ _ Value (Var example_p))).
      split; [ reflexivity | exact Hkind ].
    + right. apply example_closure_type_satisfied;
        [ reflexivity | reflexivity | exact Hkind ].
    + right. apply (proj2 (alias_satisfied _ _ _ Value (Var example_p))).
      split; [ reflexivity | exact Hkind ].
    + exact Hp.
  - apply aliases_respected_of_bindings.
    + intros n c Hin. simpl in Hin.
      repeat (destruct Hin as [Hin | Hin]; [ injection Hin as <- <- | ]);
        [ .. | contradiction ]; reflexivity.
    + intros c aliases n mode Hin Hmode. simpl in Hin.
      repeat (destruct Hin as [Hin | Hin]; [ injection Hin as <- <- | ]);
        [ .. | contradiction ]; simpl in Hmode.
      all: repeat (destruct Hmode as [Hmode | Hmode];
                   [ injection Hmode as <- <- | ]);
        [ .. | contradiction ]; reflexivity.
    + intros k aliases n mode Hin. destruct Hin.
Qed.

(* The machine at the last line binds the four variables. *)
Lemma example_env_bound (tenv : Typing_env.t) (vp vy : value) :
  example_env = Some tenv ->
  bound_in_terms tenv (example_vars vp vy).
Proof.
  intros Henv x Hscope. vm_compute in Henv. injection Henv as <-.
  apply in_scope_in_terms_lookup in Hscope.
  destruct Hscope as [ty [btm [Hlookup _]]].
  apply (keyed_lookup_in name_eqb name_eqb_eq) in Hlookup.
  simpl in Hlookup.
  repeat (destruct Hlookup as [Hlookup | Hlookup];
          [ injection Hlookup as <- <- <- | ]); [ .. | contradiction ];
    eexists; reflexivity.
Qed.

Theorem example_env_built : example_env_built_statement.
Proof. eexists. vm_compute. reflexivity. Qed.

Lemma tagged_7_or_8 (statics : static_env) (vars : env) :
  type_satisfied statics vars (tagged 7) (these_tagged_immediates [7; 8]).
Proof.
  apply these_tagged_immediates_satisfied. exists 7.
  split; [ reflexivity | left; reflexivity ].
Qed.

Theorem example_env_satisfied : example_env_satisfied_statement.
Proof.
  intros tenv Henv. apply satisfied_strictly_env_satisfies.
  - exact (example_env_bound tenv _ _ Henv).
  - apply example_env_typing_env_satisfied; [ exact Henv | ].
    right. apply tagged_7_or_8.
Qed.

Theorem example_env_poison : example_env_poison_statement.
Proof.
  intros tenv Henv. apply typing_env_satisfied_env_satisfies.
  - exact (example_env_bound tenv _ _ Henv).
  - unfold typing_env_satisfied.
    apply example_env_typing_env_satisfied; [ exact Henv | ].
    left. split; [ | reflexivity ].
    left. exists (poison Value). split; [ reflexivity | ].
    exists Value. reflexivity.
Qed.

(* A value that is not the tagged immediate 7 or 8 has not the type of
   [p]. *)
Lemma not_tagged_7_or_8 (statics : static_env) (vars : env) (v : value) :
  v <> tagged 7 -> v <> tagged 8 ->
  ~ type_satisfied statics vars v (these_tagged_immediates [7; 8]).
Proof.
  intros H7 H8 Hty.
  apply these_tagged_immediates_satisfied in Hty.
  destruct Hty as [j [-> [<- | [<- | []]]]]; [ apply H7 | apply H8 ];
    reflexivity.
Qed.

Theorem example_env_poison_not_strictly
    : example_env_poison_not_strictly_statement.
Proof.
  intros tenv Henv Hsatisfies. vm_compute in Henv. injection Henv as <-.
  apply (not_tagged_7_or_8 empty_static_env
           (example_vars (poison Value) (poison Value)) (poison Value));
    [ discriminate | discriminate | ].
  apply (find_variable_in_terms_strictly _ _ _ example_p None _ _ Hsatisfies
           eq_refl eq_refl); [ intros y [] | reflexivity ].
Qed.

Theorem example_env_poison_alias_type
    : example_env_poison_alias_type_statement.
Proof.
  intros tenv Henv. vm_compute in Henv. injection Henv as <-.
  split; [ | split ].
  - apply (proj2 (alias_satisfied _ _ _ Value (Var example_p))).
    split; reflexivity.
  - reflexivity.
  - apply not_tagged_7_or_8; discriminate.
Qed.

Theorem example_env_wrong_type : example_env_wrong_type_statement.
Proof.
  intros tenv Henv Hsatisfies. vm_compute in Henv. injection Henv as <-.
  apply (not_tagged_7_or_8 empty_static_env
           (example_vars (tagged 9) (tagged 9)) (tagged 9));
    [ discriminate | discriminate | ].
  apply (find_variable_in_terms _ _ _ example_p None _ _ Hsatisfies eq_refl
           eq_refl); [ intros y [] | reflexivity | | ].
  - intros [k Hpoison]. discriminate.
  - intros [head [relation [scrutinee [Hty _]]]]. discriminate.
Qed.

Theorem example_env_canonical : example_env_canonical_statement.
Proof.
  intros tenv Henv. vm_compute in Henv. injection Henv as <-.
  vm_compute. reflexivity.
Qed.

(* The machine binds the two variables of the environment with the
   relation. *)
Lemma example_relation_env_bound (vi vp : value) :
  bound_in_terms example_relation_env [(example_i, vi); (example_p, vp)].
Proof.
  intros x Hscope. apply in_scope_in_terms_lookup in Hscope.
  destruct Hscope as [ty [btm [Hlookup _]]].
  apply (keyed_lookup_in name_eqb name_eqb_eq) in Hlookup.
  vm_compute in Hlookup.
  repeat (destruct Hlookup as [Hlookup | Hlookup];
          [ injection Hlookup as <- <- <- | ]); [ .. | contradiction ];
    eexists; reflexivity.
Qed.

Theorem example_relation_env_immediate
    : example_relation_env_immediate_statement.
Proof.
  apply satisfied_strictly_env_satisfies;
    [ apply example_relation_env_bound | ].
  split; [ reflexivity | ]. split; [ | apply empty_respected ].
  intros n ty btm v Hlookup Hvalue.
  apply (keyed_lookup_in name_eqb name_eqb_eq) in Hlookup.
  vm_compute in Hlookup.
  repeat (destruct Hlookup as [Hlookup | Hlookup];
          [ injection Hlookup as <- <- <- | ]); [ .. | contradiction ].
  all: cbv in Hvalue; injection Hvalue as <-; right.
  all: try (first
              [ apply (proj2 (unknown_satisfied _ _ _ Value))
              | apply (proj2
                         (unknown_satisfied _ _ _
                            (Naked_number Kind_naked_immediate))) ];
            reflexivity).
  apply (proj2 (is_int_for_scrutinee_satisfied _ _ _ (Name_var example_p))).
  exists 1, (tagged 5). split; [ reflexivity | ].
  split; [ reflexivity | ]. left.
  split; [ exists 5; reflexivity | reflexivity ].
Qed.

Theorem example_relation_env_poison : example_relation_env_poison_statement.
Proof.
  apply typing_env_satisfied_env_satisfies;
    [ apply example_relation_env_bound | ].
  split; [ reflexivity | ]. split; [ | apply empty_respected ].
  intros n ty btm v Hlookup Hvalue.
  apply (keyed_lookup_in name_eqb name_eqb_eq) in Hlookup.
  vm_compute in Hlookup.
  repeat (destruct Hlookup as [Hlookup | Hlookup];
          [ injection Hlookup as <- <- <- | ]); [ .. | contradiction ].
  all: cbv in Hvalue; injection Hvalue as <-.
  all: try (right;
            first
              [ apply (proj2 (unknown_satisfied _ _ _ Value))
              | apply (proj2
                         (unknown_satisfied _ _ _
                            (Naked_number Kind_naked_immediate))) ];
            reflexivity).
  (* [i] is exempt: its type is a relation on [p], which holds poison. *)
  left. split; [ | reflexivity ].
  right. eexists. eexists. split; [ reflexivity | ].
  eexists. exists Relation.Is_int, (Name_var example_p).
  split; [ reflexivity | ]. split; [ left; reflexivity | ].
  split; [ left; reflexivity | ].
  exists (poison Value). split; [ reflexivity | ].
  exists Value. reflexivity.
Qed.

(* Where the environment is satisfied, [y] and [p] hold the same. *)
Lemma example_env_alias (tenv : Typing_env.t) (vp vy : value) :
  example_env = Some tenv ->
  env_satisfies tenv empty_static_env (example_vars vp vy) ->
  vp = vy.
Proof.
  intros Henv Hsatisfies.
  pose proof (example_env_canonical tenv Henv) as Hcanonical.
  assert (Hmem : Typing_env.mem_simple (Some Name_mode.normal) tenv
                   (Var example_y) = true).
  { vm_compute in Henv. injection Henv as <-. reflexivity. }
  pose proof (type_simple_in_term_canonical tenv empty_static_env
                (example_vars vp vy) (Var example_y) (Var example_p) _
                Hsatisfies Hmem Hcanonical) as Hvalues.
  injection Hvalues as ->. reflexivity.
Qed.

Theorem example_env_wrong_alias : example_env_wrong_alias_statement.
Proof.
  intros tenv Henv Hsatisfies.
  pose proof (example_env_alias tenv _ _ Henv Hsatisfies). discriminate.
Qed.

Theorem example_env_poison_alias : example_env_poison_alias_statement.
Proof.
  intros tenv Henv Hsatisfies.
  pose proof (example_env_alias tenv _ _ Henv Hsatisfies). discriminate.
Qed.

Theorem example_env_unbound : example_env_unbound_statement.
Proof.
  intros tenv Henv Hsatisfies. vm_compute in Henv. injection Henv as <-.
  destruct (in_scope_in_terms_bound _ _ _ example_y _ Hsatisfies eq_refl
              eq_refl) as [v [Hvalue _]].
  discriminate.
Qed.

Theorem example_env_poison_kind : example_env_poison_kind_statement.
Proof.
  intros tenv v Henv -> Hsatisfies. vm_compute in Henv.
  injection Henv as <-.
  destruct (in_scope_in_terms_bound _ _ _ example_p _ Hsatisfies eq_refl
              eq_refl) as [v [[= <-] Hkind]].
  discriminate.
Qed.

(* ======================================================================== *)
(* A name that has no value at run time                                     *)
(* ======================================================================== *)

Theorem example_body_env_built : example_body_env_built_statement.
Proof. eexists. vm_compute. reflexivity. Qed.

Theorem example_body_env_scope : example_body_env_scope_statement.
Proof.
  intros tenv Henv. vm_compute in Henv. injection Henv as <-.
  split; [ reflexivity | ]. split; [ discriminate | reflexivity ].
Qed.

(* The valuation that satisfies the environment of the body: the
   variables that are out of scope have the values they had where the
   closure was made. *)
Definition example_body_valuation (v : value) : env :=
  (example_my_closure, Value_closure example_f [(example_w, v)])
  :: example_vars v v.

Theorem example_body_env_satisfied : example_body_env_satisfied_statement.
Proof.
  intros tenv Henv. vm_compute in Henv. injection Henv as <-. split.
  - (* Only the closure called is in scope, and the machine binds it. *)
    intros x Hscope. apply in_scope_in_terms_lookup in Hscope.
    destruct Hscope as [ty [btm [Hlookup Hmode]]].
    apply (keyed_lookup_in name_eqb name_eqb_eq) in Hlookup.
    simpl in Hlookup.
    repeat (destruct Hlookup as [Hlookup | Hlookup];
            [ injection Hlookup as <- <- <- | ]); [ .. | contradiction ];
      first [ discriminate | eexists; reflexivity ].
  - exists (example_body_valuation (tagged 7)). split.
    + split; [ reflexivity | ]. split.
      * intros n ty btm v Hlookup Hvalue.
        apply (keyed_lookup_in name_eqb name_eqb_eq) in Hlookup.
        simpl in Hlookup.
        repeat (destruct Hlookup as [Hlookup | Hlookup];
                [ injection Hlookup as <- <- <- | ]); [ .. | contradiction ].
        all: cbv in Hvalue; injection Hvalue as <-.
        all: try (right; apply (proj2 (unknown_satisfied _ _ _ Value));
                  reflexivity).
        -- right. apply (proj2 (alias_satisfied _ _ _ Value (Var example_c))).
           split; reflexivity.
        -- right. apply (proj2 (alias_satisfied _ _ _ Value (Var example_p))).
           split; reflexivity.
        -- right. apply example_closure_type_satisfied; reflexivity.
        -- right. apply (proj2 (alias_satisfied _ _ _ Value (Var example_p))).
           split; reflexivity.
        -- right. apply tagged_7_or_8.
      * apply aliases_respected_of_bindings.
        -- intros n c Hin. simpl in Hin.
           repeat (destruct Hin as [Hin | Hin]; [ injection Hin as <- <- | ]);
             [ .. | contradiction ]; reflexivity.
        -- intros c aliases n mode Hin Hmode. simpl in Hin.
           repeat (destruct Hin as [Hin | Hin]; [ injection Hin as <- <- | ]);
             [ .. | contradiction ]; simpl in Hmode.
           all: repeat (destruct Hmode as [Hmode | Hmode];
                        [ injection Hmode as <- <- | ]);
             [ .. | contradiction ]; reflexivity.
        -- intros k aliases n mode Hin. destruct Hin.
    + (* The valuation agrees with the machine on the closure called. *)
      intros x Hscope. apply in_scope_in_terms_lookup in Hscope.
      destruct Hscope as [ty [btm [Hlookup Hmode]]].
      apply (keyed_lookup_in name_eqb name_eqb_eq) in Hlookup.
      simpl in Hlookup.
      repeat (destruct Hlookup as [Hlookup | Hlookup];
              [ injection Hlookup as <- <- <- | ]); [ .. | contradiction ];
        first [ reflexivity | discriminate ].
Qed.

Theorem example_body_env_wrong_slot : example_body_env_wrong_slot_statement.
Proof.
  intros tenv Henv [_ [valuation [Hsatisfied Hagrees]]].
  vm_compute in Henv. injection Henv as <-.
  (* The valuation gives the closure called the value that the machine
     has for it... *)
  pose proof (Hagrees example_my_closure eq_refl) as Hclosure.
  cbv -[lookup] in Hclosure. cbn in Hclosure.
  (* ... which is the value of [c], its canonical element... *)
  pose proof (canonical_ignoring_name_mode_satisfied _ _ valuation _
                (Var example_my_closure) Hsatisfied) as Hc.
  cbv -[lookup] in Hc. rewrite Hclosure in Hc.
  (* ... and so has the type of [c], which makes [p] the contents of
     its value slot. *)
  assert (Hty :
    type_satisfied empty_static_env valuation
      (Value_closure example_f [(example_w, tagged 9)])
      example_closure_type).
  { apply (find_not_poison empty_static_env valuation _ (Name_var example_c)
             None _ _ Hsatisfied eq_refl); [ exact Hc | | ].
    - intros [k Hpoison]. discriminate.
    - intros [head [relation [scrutinee [Hty _]]]]. discriminate. }
  apply exactly_this_closure_satisfied in Hty.
  destruct Hty as [functions [value_slots [[= <- <-] [_ [_ Hentry]]]]].
  apply closures_entry_lookup_iff in Hentry.
  destruct Hentry as [_ [_ Hslots]].
  specialize (Hslots example_w _ (tagged 9) eq_refl eq_refl).
  apply (proj1 (alias_satisfied _ _ _ Value (Var example_p))) in Hslots.
  destruct Hslots as [Hp _].
  (* But [p] is 7 or 8. *)
  apply (not_tagged_7_or_8 empty_static_env valuation (tagged 9));
    [ discriminate | discriminate | ].
  apply (find_not_poison empty_static_env valuation _ (Name_var example_p)
           None _ _ Hsatisfied eq_refl); [ exact Hp | | ].
  - intros [k Hpoison]. discriminate.
  - intros [head [relation [scrutinee [Hty _]]]]. discriminate.
Qed.

(* ======================================================================== *)
(* A variable that stands for nothing in a run                              *)
(* ======================================================================== *)

(* The immediate 0 has the type of [q], whatever [e] is. *)
Lemma example_join_type_immediate (vars : env) :
  type_satisfied empty_static_env vars (tagged 0) example_join_type.
Proof.
  apply create_variant_satisfied. left. exists 0.
  split; [ reflexivity | ]. split; [ | exact I ].
  apply (proj2 (these_naked_immediates_satisfied _ _ _ [0])). exists 0.
  split; [ reflexivity | left; reflexivity ].
Qed.

(* A block has the type of [q] when it has the type of the blocks of
   tag 0 whose one field is [e]. *)
Lemma example_join_type_block (vars : env) (v : value) :
  type_satisfied empty_static_env vars (Value_block 0 [v]) example_join_type
  <-> type_satisfied empty_static_env vars (Value_block 0 [v])
        (immutable_block false 0 (Scannable Value_only) Alloc_mode.Heap
           [alias_type_of Value (Var example_e)]).
Proof.
  unfold example_join_type, immutable_block. split; intros Hty.
  - apply create_variant_satisfied in Hty. apply create_variant_satisfied.
    destruct Hty as [[i [Hi _]] | Hblock];
      [ discriminate | right; exact Hblock ].
  - apply create_variant_satisfied in Hty. apply create_variant_satisfied.
    destruct Hty as [[i [Hi _]] | Hblock];
      [ discriminate | right; exact Hblock ].
Qed.

(* Only [q] is in scope. *)
Lemma example_join_env_scope (x : variable) :
  in_scope_in_terms example_join_env x -> x = example_q.
Proof.
  intros Hscope. apply in_scope_in_terms_lookup in Hscope.
  destruct Hscope as [ty [btm [Hlookup Hmode]]].
  apply (keyed_lookup_in name_eqb name_eqb_eq) in Hlookup.
  vm_compute in Hlookup.
  repeat (destruct Hlookup as [Hlookup | Hlookup];
          [ injection Hlookup as <- <- <- | ]); [ .. | contradiction ];
    first [ reflexivity | discriminate ].
Qed.

Theorem example_join_env_immediate : example_join_env_immediate_statement.
Proof.
  apply satisfied_strictly_env_satisfies.
  - intros x Hscope. apply example_join_env_scope in Hscope. subst x.
    eexists. reflexivity.
  - split; [ reflexivity | ]. split; [ | apply empty_respected ].
    intros n ty btm v Hlookup Hvalue.
    apply (keyed_lookup_in name_eqb name_eqb_eq) in Hlookup.
    vm_compute in Hlookup.
    repeat (destruct Hlookup as [Hlookup | Hlookup];
            [ injection Hlookup as <- <- <- | ]); [ .. | contradiction ].
    all: cbv in Hvalue; try discriminate; injection Hvalue as <-; right.
    all: try (apply (proj2 (unknown_satisfied _ _ _ Value)); reflexivity).
    apply example_join_type_immediate.
Qed.

Theorem example_join_env_block : example_join_env_block_statement.
Proof.
  split.
  - intros x Hscope. apply example_join_env_scope in Hscope. subst x.
    eexists. reflexivity.
  - exists [(example_e, tagged 4); (example_q, Value_block 0 [tagged 4])].
    split.
    + split; [ reflexivity | ]. split; [ | apply empty_respected ].
      intros n ty btm v Hlookup Hvalue.
      apply (keyed_lookup_in name_eqb name_eqb_eq) in Hlookup.
      vm_compute in Hlookup.
      repeat (destruct Hlookup as [Hlookup | Hlookup];
              [ injection Hlookup as <- <- <- | ]); [ .. | contradiction ].
      all: cbv in Hvalue; injection Hvalue as <-; right.
      all: try (apply (proj2 (unknown_satisfied _ _ _ Value)); reflexivity).
      * apply (proj2 (example_join_type_block _ (tagged 4))).
        apply immutable_block_satisfied. exists [tagged 4].
        split; [ reflexivity | ]. split; [ split; reflexivity | ].
        constructor; [ | constructor ].
        apply (proj2 (alias_satisfied _ _ _ Value (Var example_e))).
        split; reflexivity.
      * apply (proj2 (these_tagged_immediates_satisfied _ _ _ [4; 5])).
        exists 4. split; [ reflexivity | left; reflexivity ].
    + intros x Hscope. apply example_join_env_scope in Hscope. subst x.
      reflexivity.
Qed.

Theorem example_join_env_wrong_block
    : example_join_env_wrong_block_statement.
Proof.
  intros [_ [valuation [Hsatisfied Hagrees]]].
  (* The valuation gives [q] the block that the machine has... *)
  pose proof (Hagrees example_q eq_refl) as Hq.
  cbv -[lookup] in Hq. cbn in Hq.
  (* ... which has the type of [q], so that [e] is its field... *)
  assert (Hty :
    type_satisfied empty_static_env valuation (Value_block 0 [tagged 6])
      example_join_type).
  { apply (find_not_poison empty_static_env valuation example_join_env
             (Name_var example_q) None _ _ Hsatisfied eq_refl);
      [ exact Hq | | ].
    - intros [k Hpoison]. discriminate.
    - intros [head [relation [scrutinee [Hty _]]]]. discriminate. }
  apply example_join_type_block in Hty.
  apply immutable_block_satisfied in Hty.
  destruct Hty as [fields [[= <-] [_ Hfields]]].
  inversion Hfields as [ | field ty fields' tys Halias _ ]; subst.
  apply (proj1 (alias_satisfied _ _ _ Value (Var example_e))) in Halias.
  destruct Halias as [He _].
  (* ... and [e] is 4. *)
  assert (Hty :
    type_satisfied empty_static_env valuation (tagged 6)
      (these_tagged_immediates [4; 5])).
  { apply (find_not_poison empty_static_env valuation example_join_env
             (Name_var example_e) None _ _ Hsatisfied eq_refl);
      [ exact He | | ].
    - intros [k Hpoison]. discriminate.
    - intros [head [relation [scrutinee [Hty _]]]]. discriminate. }
  apply these_tagged_immediates_satisfied in Hty.
  destruct Hty as [i [[= <-] [Hi | [Hi | []]]]]; discriminate.
Qed.

Theorem example_absent_env_satisfied
    : example_absent_env_satisfied_statement.
Proof.
  intros statics valuation. split.
  - intros Hsatisfied.
    destruct (lookup example_e valuation) as [v | ] eqn:Hvalue;
      [ exfalso | reflexivity ].
    apply (bottom_satisfied statics valuation v Value).
    exact (find_satisfied_strictly statics valuation example_absent_env
             (Name_var example_e) None (bottom Value) v Hsatisfied eq_refl
             Hvalue).
  - intros Hvalue. split; [ reflexivity | ].
    split; [ | apply empty_respected ].
    intros n ty btm v Hlookup Hv.
    apply (keyed_lookup_in name_eqb name_eqb_eq) in Hlookup.
    vm_compute in Hlookup.
    repeat (destruct Hlookup as [Hlookup | Hlookup];
            [ injection Hlookup as <- <- <- | ]); [ .. | contradiction ].
    all: change (lookup example_e valuation = Some v) in Hv; congruence.
Qed.

(* ======================================================================== *)
(* A symbol                                                                 *)
(* ======================================================================== *)

Theorem example_defined_symbol : example_defined_symbol_statement.
Proof.
  intros statics vars. apply satisfied_strictly_env_satisfies.
  - intros x Hscope. discriminate.
  - split; [ reflexivity | ]. split.
    + intros n ty btm v Hlookup. discriminate.
    + apply empty_respected.
Qed.

Theorem example_symbol_env_satisfied : example_symbol_env_satisfied_statement.
Proof.
  apply satisfied_strictly_env_satisfies.
  - intros x Hscope. discriminate.
  - split; [ reflexivity | ]. split.
    + intros n ty btm v Hlookup Hvalue.
      apply (keyed_lookup_in name_eqb name_eqb_eq) in Hlookup.
      destruct Hlookup as [[= <- <- <-] | []].
      injection Hvalue as <-.
      right. apply immutable_block_satisfied. exists [tagged 1].
      split; [ reflexivity | ]. split; [ split; reflexivity | ].
      constructor; [ | constructor ].
      apply (proj2 (these_tagged_immediates_satisfied _ _ _ [1])). exists 1.
      split; [ reflexivity | left; reflexivity ].
    + apply empty_respected.
Qed.

Theorem example_symbol_env_unbound : example_symbol_env_unbound_statement.
Proof.
  intros vars Hsatisfies.
  pose proof (find_symbol_in_terms example_symbol_env empty_static_env vars
                example_s None _ Hsatisfies eq_refl) as Hty.
  cbv beta in Hty.
  assert (Hblock :
    type_satisfied empty_static_env vars (Value_symbol example_s)
      (immutable_block false 0 (Scannable Value_only) Alloc_mode.Heap
         [these_tagged_immediates [1]])).
  { apply Hty. intros y []. }
  apply immutable_block_satisfied in Hblock.
  destruct Hblock as [fields [Hblock _]]. discriminate.
Qed.
