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


(* The functions with which the meet of a type is put together from
   the meets of its parts commute with a renaming of variables. Lemmas
   only: nothing here is a statement to review.

   For each function of FlambdaMeet.v from [map_return_value] to
   [meet_inverse_relations], a lemma named after the function says
   that on the renamed environment and renamed arguments it gives the
   renamed answer. The action on what a meet takes and answers is that
   of FlambdaMeetRenameSpec.v. A function that is given the meet of a
   part is given, on the renamed side, a meet that answers the renamed
   answer of renamed parts; the two may be one function. The keys of a
   map and the numbers of a set are not renamed: they are tags, slots
   and numbers. The environment of the meet is
   FlambdaEnvRenameMeetProofs.v. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaMapProofs FlambdaNominalSpec FlambdaNominalProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeRenameSpec FlambdaTypeRenameProofs FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesRenamingSpec
  FlambdaEnvAliasesRenamingProofs FlambdaTypingEnv FlambdaEnvRenameSpec
  FlambdaEnvRenameProofs FlambdaMeetGrammar FlambdaExpandHead
  FlambdaProversRenameSpec FlambdaMeetEnv FlambdaMeet FlambdaMeetRenameSpec
  FlambdaEnvRenameMeetProofs.
Import ListNotations.

(* ======================================================================== *)
(* What a meet answers with                                                 *)
(* ======================================================================== *)

Lemma map_return_value_renamed {A B : Type} (f : A -> A) (g : B -> B)
    (h h' : A -> B) (x : meet_return_value A) :
  (∀ a : A, h' (f a) = g (h a)) ->
  map_return_value h' (rename_return_value f x)
  = rename_return_value g (map_return_value h x).
Proof.
  intros Hh. destruct x as [ | | | a];
    cbn [map_return_value rename_return_value];
    try reflexivity.
  rewrite Hh. reflexivity.
Qed.

(* The value that is computed for a new result is, on the renamed
   side, the renamed value. *)
Lemma combine_meet_return_values_renamed {A B C : Type} (fa : A -> A)
    (fb : B -> B) (fc : C -> C) (a : meet_return_value A)
    (b : meet_return_value B) (compute_value compute_value' : unit -> C) :
  compute_value' tt = fc (compute_value tt) ->
  combine_meet_return_values (rename_return_value fa a)
    (rename_return_value fb b) compute_value'
  = rename_return_value fc (combine_meet_return_values a b compute_value).
Proof.
  intros Hcompute.
  destruct a, b;
    cbn [combine_meet_return_values rename_return_value];
    first [ reflexivity | rewrite Hcompute; reflexivity ].
Qed.

Lemma sequence_return_value_renamed {A : Type} (f : A -> A)
    (x : meet_return_value (option A)) :
  sequence_return_value (rename_return_value (option_map f) x)
  = option_map (rename_return_value f) (sequence_return_value x).
Proof. destruct x as [ | | | [a | ]]; reflexivity. Qed.

(* A set of numbers: what is made of the set is left as it is. *)
Lemma set_meet_renamed {K B : Type} (r : renaming) (eqb : K -> K -> bool)
    (of_set : list K -> B) (g : B -> B) (env : Meet_env.t)
    (s1 s2 : list K) :
  (∀ s : list K, g (of_set s) = of_set s) ->
  set_meet eqb of_set (rename_meet_env r env) s1 s2
  = rename_meet_result r g (set_meet eqb of_set env s1 s2).
Proof.
  intros Hg. unfold set_meet.
  destruct (Keyed.all_among eqb s1 s2), (Keyed.all_among eqb s2 s1);
    try reflexivity.
  cbv zeta.
  destruct (Keyed.is_empty (Keyed.inter eqb s1 s2)); [ reflexivity | ].
  cbn [rename_meet_result rename_return_value]. rewrite Hg. reflexivity.
Qed.

(* ======================================================================== *)
(* The meet of two products                                                 *)
(* ======================================================================== *)

(* The state of [meet_mapping], renamed: the components and the
   environment are; the keys are not. *)
Definition rename_mapping_acc {K A : Type} (r : renaming) (f : A -> A)
    (acc : mapping_acc K A) : mapping_acc K A :=
  {| all_left := all_left acc;
     all_right := all_right acc;
     mapping := Keyed.map_values f (mapping acc);
     mapping_env := rename_meet_env r (mapping_env acc) |}.

Definition rename_mapping_state {K A : Type} (r : renaming) (f : A -> A)
    (state : mapping_state K A) : mapping_state K A :=
  option_map (or_bottom_map (rename_mapping_acc r f)) state.

(* [f] renames a component and [g] a product. The fold is to go
   through the renamed products as it goes through the products, and
   the product made of renamed components is to be the renamed
   product. *)
Lemma meet_mapping_renamed {K A M : Type} (r : renaming) (f : A -> A)
    (g : M -> M)
    (meet_data meet_data' : Meet_env.t -> A -> A -> option (Meet_result.t A))
    (fold2 fold2' :
      (K -> option A -> option A -> mapping_state K A -> mapping_state K A) ->
      M -> M -> mapping_state K A -> mapping_state K A)
    (env : Meet_env.t) (left right : M)
    (rebuild rebuild' : list (K * A) -> M) :
  (∀ (env : Meet_env.t) (a1 a2 : A),
    meet_data' (rename_meet_env r env) (f a1) (f a2)
    = option_map (rename_meet_result r f) (meet_data env a1 a2)) ->
  (∀ (step step' :
        K -> option A -> option A -> mapping_state K A -> mapping_state K A)
     (init : mapping_state K A),
    (∀ (key : K) (data_left data_right : option A)
       (state : mapping_state K A),
      step' key (option_map f data_left) (option_map f data_right)
        (rename_mapping_state r f state)
      = rename_mapping_state r f (step key data_left data_right state)) ->
    fold2' step' (g left) (g right) (rename_mapping_state r f init)
    = rename_mapping_state r f (fold2 step left right init)) ->
  (∀ l : list (K * A), rebuild' (Keyed.map_values f l) = g (rebuild l)) ->
  meet_mapping meet_data' fold2' (rename_meet_env r env) (g left) (g right)
    rebuild'
  = option_map (rename_meet_result r g)
      (meet_mapping meet_data fold2 env left right rebuild).
Proof.
  intros Hmeet Hfold2 Hrebuild. unfold meet_mapping. cbv zeta.
  match goal with
  | |- match fold2' ?step' _ _ ?init' with _ => _ end
       = option_map _ (match fold2 ?step _ _ ?init with _ => _ end) =>
    change init' with (rename_mapping_state r f init);
    rewrite (Hfold2 step step' init);
    [ destruct (fold2 step left right init) as [[acc | ] | ] | ]
  end.
  - cbn [rename_mapping_state option_map or_bottom_map rename_mapping_acc
         all_left all_right mapping mapping_env].
    destruct (all_left acc), (all_right acc);
      cbn [option_map rename_meet_result rename_return_value];
      try reflexivity.
    rewrite Hrebuild. reflexivity.
  - reflexivity.
  - reflexivity.
  - intros key [data_left | ] [data_right | ] [[acc | ] | ];
      cbn [rename_mapping_state option_map or_bottom_map rename_mapping_acc
           all_left all_right mapping mapping_env];
      try reflexivity.
    rewrite Hmeet.
    destruct (meet_data (mapping_env acc) data_left data_right)
      as [[res | res env'] | ];
      cbn [option_map rename_meet_result]; try reflexivity.
    destruct res; reflexivity.
Qed.

(* [Map_meet.meet]: the maps are renamed by what their keys are bound
   to. *)
Lemma map_meet_renamed {K A : Type} (r : renaming) (eqb : K -> K -> bool)
    (f : A -> A)
    (meet_data meet_data' : Meet_env.t -> A -> A -> option (Meet_result.t A))
    (env : Meet_env.t) (left right : Keyed.finite_map K A) :
  (∀ (env : Meet_env.t) (a1 a2 : A),
    meet_data' (rename_meet_env r env) (f a1) (f a2)
    = option_map (rename_meet_result r f) (meet_data env a1 a2)) ->
  map_meet eqb meet_data' (rename_meet_env r env) (Keyed.map_values f left)
    (Keyed.map_values f right)
  = option_map (rename_meet_result r (Keyed.map_values f))
      (map_meet eqb meet_data env left right).
Proof.
  intros Hmeet. unfold map_meet. cbv zeta.
  apply (meet_mapping_renamed r f (Keyed.map_values f)); [ exact Hmeet | | ].
  - intros step step' init Hstep.
    assert (Hkeys :
      Keyed.keys_of_both eqb (Keyed.map_values f left)
        (Keyed.map_values f right)
      = Keyed.keys_of_both eqb left right).
    { unfold Keyed.keys_of_both, Keyed.map_values. rewrite !map_map.
      reflexivity. }
    rewrite Hkeys. clear Hkeys. unfold Keyed.fold2_on. revert init.
    induction (Keyed.keys_of_both eqb left right) as [ | key keys IH];
      intros init; cbn [List.fold_left]; [ reflexivity | ].
    rewrite !keyed_lookup_map_values, Hstep. apply IH.
  - intros l.
    exact
      (fold_left_renamed (Keyed.map_values f)
         (fun binding => (fst binding, f (snd binding)))
         (fun m binding => binding :: m) (fun m binding => binding :: m) l
         (fun m binding => eq_refl) []).
Qed.

(* ======================================================================== *)
(* Meets made one after the other                                           *)
(* ======================================================================== *)

Lemma combine_results2_renamed {A B C : Type} (r : renaming) (fa : A -> A)
    (fb : B -> B) (fc : C -> C) (env : Meet_env.t)
    (meet_a meet_a' : Meet_env.t -> A -> A -> option (Meet_result.t A))
    (meet_b meet_b' : Meet_env.t -> B -> B -> option (Meet_result.t B))
    (left_a right_a : A) (left_b right_b : B)
    (rebuild rebuild' : A -> B -> C) :
  (∀ (env : Meet_env.t) (a1 a2 : A),
    meet_a' (rename_meet_env r env) (fa a1) (fa a2)
    = option_map (rename_meet_result r fa) (meet_a env a1 a2)) ->
  (∀ (env : Meet_env.t) (b1 b2 : B),
    meet_b' (rename_meet_env r env) (fb b1) (fb b2)
    = option_map (rename_meet_result r fb) (meet_b env b1 b2)) ->
  (∀ (a : A) (b : B), rebuild' (fa a) (fb b) = fc (rebuild a b)) ->
  combine_results2 (rename_meet_env r env) meet_a' meet_b' (fa left_a)
    (fa right_a) (fb left_b) (fb right_b) rebuild'
  = option_map (rename_meet_result r fc)
      (combine_results2 env meet_a meet_b left_a right_a left_b right_b
         rebuild).
Proof.
  intros Ha Hb Hrebuild. unfold combine_results2. rewrite Ha.
  destruct (meet_a env left_a right_a) as [[res_a | res_a env_a] | ];
    cbn [option_map rename_meet_result]; try reflexivity.
  rewrite Hb.
  destruct (meet_b env_a left_b right_b) as [[res_b | res_b env_b] | ];
    cbn [option_map rename_meet_result]; try reflexivity.
  cbv zeta.
  destruct res_a, res_b;
    cbn [combine_meet_return_values extract_value rename_meet_result
         rename_return_value option_map];
    first [ reflexivity | rewrite Hrebuild; reflexivity ].
Qed.

Lemma combine_results3_renamed {A B C D : Type} (r : renaming)
    (fa : A -> A) (fb : B -> B) (fc : C -> C) (fd : D -> D)
    (env : Meet_env.t)
    (meet_a meet_a' : Meet_env.t -> A -> A -> option (Meet_result.t A))
    (meet_b meet_b' : Meet_env.t -> B -> B -> option (Meet_result.t B))
    (meet_c meet_c' : Meet_env.t -> C -> C -> option (Meet_result.t C))
    (left_a right_a : A) (left_b right_b : B) (left_c right_c : C)
    (rebuild rebuild' : A -> B -> C -> D) :
  (∀ (env : Meet_env.t) (a1 a2 : A),
    meet_a' (rename_meet_env r env) (fa a1) (fa a2)
    = option_map (rename_meet_result r fa) (meet_a env a1 a2)) ->
  (∀ (env : Meet_env.t) (b1 b2 : B),
    meet_b' (rename_meet_env r env) (fb b1) (fb b2)
    = option_map (rename_meet_result r fb) (meet_b env b1 b2)) ->
  (∀ (env : Meet_env.t) (c1 c2 : C),
    meet_c' (rename_meet_env r env) (fc c1) (fc c2)
    = option_map (rename_meet_result r fc) (meet_c env c1 c2)) ->
  (∀ (a : A) (b : B) (c : C),
    rebuild' (fa a) (fb b) (fc c) = fd (rebuild a b c)) ->
  combine_results3 (rename_meet_env r env) meet_a' meet_b' meet_c'
    (fa left_a) (fa right_a) (fb left_b) (fb right_b) (fc left_c)
    (fc right_c) rebuild'
  = option_map (rename_meet_result r fd)
      (combine_results3 env meet_a meet_b meet_c left_a right_a left_b
         right_b left_c right_c rebuild).
Proof.
  intros Ha Hb Hc Hrebuild. unfold combine_results3. rewrite Ha.
  destruct (meet_a env left_a right_a) as [[res_a | res_a env_a] | ];
    cbn [option_map rename_meet_result]; try reflexivity.
  rewrite Hb.
  destruct (meet_b env_a left_b right_b) as [[res_b | res_b env_b] | ];
    cbn [option_map rename_meet_result]; try reflexivity.
  rewrite Hc.
  destruct (meet_c env_b left_c right_c) as [[res_c | res_c env_c] | ];
    cbn [option_map rename_meet_result]; try reflexivity.
  cbv zeta.
  destruct res_a, res_b, res_c;
    cbn [combine_meet_return_values extract_value rename_meet_result
         rename_return_value option_map fst snd];
    first [ reflexivity | rewrite Hrebuild; reflexivity ].
Qed.

(* ======================================================================== *)
(* The meets that need no other                                             *)
(* ======================================================================== *)

(* A code id is not renamed, nor an allocation mode; whatever [g] is,
   neither meet answers a new result. *)
Lemma meet_code_id_renamed (r : renaming) (g : code_id -> code_id)
    (env : Meet_env.t) (code_id1 code_id2 : code_id) :
  meet_code_id (rename_meet_env r env) code_id1 code_id2
  = rename_meet_result r g (meet_code_id env code_id1 code_id2).
Proof.
  unfold meet_code_id. destruct (Pos.eqb code_id1 code_id2); reflexivity.
Qed.

Lemma meet_alloc_mode_renamed (r : renaming)
    (g : Alloc_mode.t -> Alloc_mode.t) (env : Meet_env.t)
    (alloc_mode1 alloc_mode2 : Alloc_mode.t) :
  meet_alloc_mode (rename_meet_env r env) alloc_mode1 alloc_mode2
  = rename_meet_result r g (meet_alloc_mode env alloc_mode1 alloc_mode2).
Proof. destruct alloc_mode1, alloc_mode2; reflexivity. Qed.

Lemma meet_unknown_renamed {A : Type} (r : renaming) (f : A -> A)
    (meet_contents meet_contents' :
      Meet_env.t -> A -> A -> option (Meet_result.t A))
    (contents_is_bottom contents_is_bottom' : A -> bool) (env : Meet_env.t)
    (or_unknown1 or_unknown2 : Or_unknown.t A) :
  (∀ (env : Meet_env.t) (a1 a2 : A),
    meet_contents' (rename_meet_env r env) (f a1) (f a2)
    = option_map (rename_meet_result r f) (meet_contents env a1 a2)) ->
  (∀ a : A, contents_is_bottom' (f a) = contents_is_bottom a) ->
  meet_unknown meet_contents' contents_is_bottom' (rename_meet_env r env)
    (or_unknown_map f or_unknown1) (or_unknown_map f or_unknown2)
  = option_map (rename_meet_result r (or_unknown_map f))
      (meet_unknown meet_contents contents_is_bottom env or_unknown1
         or_unknown2).
Proof.
  intros Hmeet Hbottom.
  destruct or_unknown1 as [contents1 | ], or_unknown2 as [contents2 | ];
    cbn [meet_unknown or_unknown_map]; rewrite ?Hbottom.
  - destruct (contents_is_bottom contents1), (contents_is_bottom contents2);
      cbn [andb]; try reflexivity.
    rewrite Hmeet.
    destruct (meet_contents env contents1 contents2) as [result | ];
      cbn [option_map]; [ | reflexivity ].
    f_equal. apply map_result_renamed. intros a. reflexivity.
  - destruct (contents_is_bottom contents1); reflexivity.
  - destruct (contents_is_bottom contents2); reflexivity.
  - reflexivity.
Qed.

Lemma meet_or_unknown_or_bottom_renamed {A B : Type} (r : renaming)
    (f : A -> A) (g : B -> B)
    (meet_elt meet_elt' : Meet_env.t -> A -> A -> option (Meet_result.t B))
    (env : Meet_env.t) (input1 input2 : or_unknown_or_bottom A) :
  (∀ (env : Meet_env.t) (a1 a2 : A),
    meet_elt' (rename_meet_env r env) (f a1) (f a2)
    = option_map (rename_meet_result r g) (meet_elt env a1 a2)) ->
  meet_or_unknown_or_bottom meet_elt' (rename_meet_env r env)
    (or_unknown_or_bottom_map f input1) (or_unknown_or_bottom_map f input2)
  = option_map (rename_meet_result r g)
      (meet_or_unknown_or_bottom meet_elt env input1 input2).
Proof.
  intros Hmeet.
  destruct input1 as [ | elt1 | ], input2 as [ | elt2 | ];
    cbn [meet_or_unknown_or_bottom or_unknown_or_bottom_map];
    try reflexivity.
  apply Hmeet.
Qed.

(* ======================================================================== *)
(* Naked immediates: the relations                                          *)
(* ======================================================================== *)

(* The relations of a head of naked immediates, renamed: the name of
   each is. This is how [rename_head_of_kind_naked_immediate] writes
   it. *)
Local Abbreviation rename_relations r :=
  (map
     (fun relation : Relation.t * name =>
        (fst relation, rename_name r (snd relation)))).

Lemma inverse_relation_eqb_renamed (r : renaming)
    (relation relation' : Relation.t * name) :
  inverse_relation_eqb (fst relation, rename_name r (snd relation))
    (fst relation', rename_name r (snd relation'))
  = inverse_relation_eqb relation relation'.
Proof.
  unfold inverse_relation_eqb. cbn [fst snd]. rewrite name_eqb_renamed.
  reflexivity.
Qed.

Lemma head_of_kind_naked_immediate_from_descr_renamed (r : renaming)
    (imms : Or_unknown.t (list Z)) (relations : list (Relation.t * name)) :
  head_of_kind_naked_immediate_from_descr imms (rename_relations r relations)
  = or_bottom_map (rename_head_of_kind_naked_immediate r)
      (head_of_kind_naked_immediate_from_descr imms relations).
Proof. destruct imms as [[ | imm imms] | ]; reflexivity. Qed.

Lemma head_of_kind_naked_immediate_from_descr_non_empty_renamed
    (r : renaming) (imms : Or_unknown.t (list Z))
    (relations : list (Relation.t * name)) :
  head_of_kind_naked_immediate_from_descr_non_empty imms
    (rename_relations r relations)
  = option_map (rename_head_of_kind_naked_immediate r)
      (head_of_kind_naked_immediate_from_descr_non_empty imms relations).
Proof. destruct imms as [[ | imm imms] | ]; reflexivity. Qed.

(* The state of [reduce_head_of_kind_naked_immediate], renamed: the
   relations kept are. *)
Definition rename_reduce_acc (r : renaming) (acc : reduce_acc) : reduce_acc :=
  {| reduce_imms := reduce_imms acc;
     reduce_relations := rename_relations r (reduce_relations acc);
     reduce_changed := reduce_changed acc |}.

Lemma reduce_head_of_kind_naked_immediate_renamed (r : renaming)
    (tenv : Typing_env.t) (head : head_of_kind_naked_immediate) :
  reduce_head_of_kind_naked_immediate (rename_typing_env r tenv)
    (rename_head_of_kind_naked_immediate r head)
  = or_bottom_map (rename_head_of_kind_naked_immediate r)
      (reduce_head_of_kind_naked_immediate tenv head).
Proof.
  unfold reduce_head_of_kind_naked_immediate. cbv zeta.
  change (inverse_relations (rename_head_of_kind_naked_immediate r head))
    with (rename_relations r (inverse_relations head)).
  change (naked_immediates (rename_head_of_kind_naked_immediate r head))
    with (naked_immediates head).
  match goal with
  | |- match List.fold_left ?step' _ ?init' with _ => _ end
       = or_bottom_map _ (match List.fold_left ?step _ ?init with _ => _ end)
    =>
    change init' with (or_bottom_map (rename_reduce_acc r) init) at 1;
    rewrite
      (fold_left_renamed (or_bottom_map (rename_reduce_acc r))
         (fun relation : Relation.t * name =>
            (fst relation, rename_name r (snd relation)))
         step step');
    [ destruct (List.fold_left step (inverse_relations head) init)
        as [acc | ]
    | ]
  end.
  - cbn [or_bottom_map rename_reduce_acc reduce_changed reduce_imms
         reduce_relations].
    destruct (reduce_changed acc); cbn [negb]; [ | reflexivity ].
    rewrite <- map_rev,
      (dedup_renamed inverse_relation_eqb _ (inverse_relation_eqb_renamed r)).
    apply head_of_kind_naked_immediate_from_descr_renamed.
  - reflexivity.
  - intros [acc | ] [relation name]; cbn [or_bottom_map fst snd];
      [ | reflexivity ].
    rewrite <- rename_simple_of_name,
      get_canonical_ignoring_name_mode_equivariant.
    destruct
      (Typing_env.get_canonical_simple_ignoring_name_mode tenv
         (simple_of_name name))
      as [x | sym | const];
      cbn [rename_simple rename_reduce_acc reduce_changed reduce_imms
           reduce_relations map fst snd].
    + change (Name_var (rename_variable r x))
        with (rename_name r (Name_var x)).
      rewrite name_eqb_renamed. reflexivity.
    + change (name_eqb (Name_symbol sym) (rename_name r name))
        with (name_eqb (rename_name r (Name_symbol sym)) (rename_name r name)).
      rewrite name_eqb_renamed. reflexivity.
    + destruct (Relation.of_const relation const) as [imm | ];
        [ | reflexivity ].
      destruct (reduce_imms acc) as [imms | ]; [ | reflexivity ].
      destruct (Keyed.same_keys_b Z.eqb [imm] imms); [ reflexivity | ].
      destruct (negb (Keyed.among Z.eqb imm imms)); reflexivity.
Qed.

(* The three parts of what [meet_inverse_relations] answers. *)
Lemma meet_inverse_relations_renamed (r : renaming)
    (relations1 relations2 : list (Relation.t * name)) :
  meet_inverse_relations (rename_relations r relations1)
    (rename_relations r relations2)
  = {| left_inputs_only :=
         rename_relations r
           (left_inputs_only (meet_inverse_relations relations1 relations2));
       meet_return_value_of :=
         rename_return_value (rename_relations r)
           (meet_return_value_of
              (meet_inverse_relations relations1 relations2));
       right_inputs_only :=
         rename_relations r
           (right_inputs_only
              (meet_inverse_relations relations1 relations2)) |}.
Proof.
  unfold meet_inverse_relations. cbv zeta.
  cbn [left_inputs_only meet_return_value_of right_inputs_only].
  rewrite
    !(without_renamed inverse_relation_eqb _ (inverse_relation_eqb_renamed r)),
    <- map_app,
    (dedup_renamed inverse_relation_eqb _ (inverse_relation_eqb_renamed r)),
    !is_empty_map.
  destruct
    (Keyed.is_empty (Keyed.without inverse_relation_eqb relations2 relations1)),
    (Keyed.is_empty (Keyed.without inverse_relation_eqb relations1 relations2));
    reflexivity.
Qed.
